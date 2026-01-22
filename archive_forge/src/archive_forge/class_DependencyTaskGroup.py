import sys
import types
import eventlet
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from heat.common.i18n import _
from heat.common import timeutils
class DependencyTaskGroup(object):
    """Task which manages group of subtasks that have ordering dependencies."""

    def __init__(self, dependencies, task=lambda o: o(), reverse=False, name=None, error_wait_time=None, aggregate_exceptions=False):
        """Initialise with the task dependencies.

        A task to run on each dependency may optionally be specified.  If no
        task is supplied, it is assumed that the tasks are stored directly in
        the dependency tree. If a task is supplied, the object stored in the
        dependency tree is passed as an argument.

        If an error_wait_time is specified, tasks that are already running at
        the time of an error will continue to run for up to the specified time
        before being cancelled. Once all remaining tasks are complete or have
        been cancelled, the original exception is raised. If error_wait_time is
        a callable function it will be called for each task, passing the
        dependency key as an argument, to determine the error_wait_time for
        that particular task.

        If aggregate_exceptions is True, then execution of parallel operations
        will not be cancelled in the event of an error (operations downstream
        of the error will be cancelled). Once all chains are complete, any
        errors will be rolled up into an ExceptionGroup exception.
        """
        self._keys = list(dependencies)
        self._runners = dict(((o, TaskRunner(task, o)) for o in self._keys))
        self._graph = dependencies.graph(reverse=reverse)
        self.error_wait_time = error_wait_time
        self.aggregate_exceptions = aggregate_exceptions
        if name is None:
            name = '(%s) %s' % (getattr(task, '__name__', task_description(task)), str(dependencies))
        self.name = name

    def __repr__(self):
        """Return a string representation of the task."""
        text = '%s(%s)' % (type(self).__name__, self.name)
        return text

    def __call__(self):
        """Return a co-routine which runs the task group."""
        raised_exceptions = []
        thrown_exceptions = []
        try:
            while any(self._runners.values()):
                try:
                    for k, r in self._ready():
                        r.start()
                        if not r:
                            del self._graph[k]
                    if self._graph:
                        try:
                            yield
                        except Exception as err:
                            thrown_exceptions.append(err)
                            raise
                    for k, r in self._running():
                        if r.step():
                            del self._graph[k]
                except Exception as err:
                    if self.aggregate_exceptions:
                        self._cancel_recursively(k, r)
                    else:
                        self.cancel_all(grace_period=self.error_wait_time)
                    raised_exceptions.append(err)
                except:
                    with excutils.save_and_reraise_exception():
                        self.cancel_all()
            if raised_exceptions:
                if self.aggregate_exceptions:
                    raise ExceptionGroup((err for err in raised_exceptions))
                elif thrown_exceptions:
                    raise thrown_exceptions[-1]
                else:
                    raise raised_exceptions[0]
        finally:
            del raised_exceptions
            del thrown_exceptions

    def cancel_all(self, grace_period=None):
        if callable(grace_period):
            get_grace_period = grace_period
        else:

            def get_grace_period(key):
                return grace_period
        for k, r in self._runners.items():
            if not r.started() or r.done():
                gp = None
            else:
                gp = get_grace_period(k)
            try:
                r.cancel(grace_period=gp)
            except Exception as ex:
                LOG.debug('Exception cancelling task: %s', str(ex))

    def _cancel_recursively(self, key, runner):
        try:
            runner.cancel()
        except Exception as ex:
            LOG.debug('Exception cancelling task: %s', str(ex))
        node = self._graph[key]
        for dependent_node in node.required_by():
            node_runner = self._runners[dependent_node]
            self._cancel_recursively(dependent_node, node_runner)
        del self._graph[key]

    def _ready(self):
        """Iterate over all subtasks that are ready to start.

        Ready subtasks are subtasks whose dependencies have all been satisfied,
        but which have not yet been started.
        """
        for k in self._keys:
            if not self._graph.get(k, True):
                runner = self._runners[k]
                if runner and (not runner.started()):
                    yield (k, runner)

    def _running(self):
        """Iterate over all subtasks that are currently running.

        Running subtasks are subtasks have been started but have not yet
        completed.
        """

        def running(k_r):
            return k_r[0] in self._graph and k_r[1].started()
        return filter(running, self._runners.items())