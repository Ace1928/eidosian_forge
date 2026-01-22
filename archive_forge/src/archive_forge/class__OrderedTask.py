import abc
import functools
import inspect
import logging
import threading
import traceback
from oslo_config import cfg
from oslo_service import service
from oslo_utils import eventletutils
from oslo_utils import timeutils
from stevedore import driver
from oslo_messaging._drivers import base as driver_base
from oslo_messaging import _utils as utils
from oslo_messaging import exceptions
class _OrderedTask(object):
    """A task which must be executed in a particular order.

    A caller may wait for this task to complete by calling
    `wait_for_completion`.

    A caller may run this task with `run_once`, which will ensure that however
    many times the task is called it only runs once. Simultaneous callers will
    block until the running task completes, which means that any caller can be
    sure that the task has completed after run_once returns.
    """
    INIT = 0
    RUNNING = 1
    COMPLETE = 2

    def __init__(self, name):
        """Create a new _OrderedTask.

        :param name: The name of this task. Used in log messages.
        """
        super(_OrderedTask, self).__init__()
        self._name = name
        self._cond = threading.Condition()
        self._state = self.INIT

    def _wait(self, condition, msg, log_after, timeout_timer):
        """Wait while condition() is true. Write a log message if condition()
        has not become false within `log_after` seconds. Raise TaskTimeout if
        timeout_timer expires while waiting.
        """
        log_timer = None
        if log_after != 0:
            log_timer = timeutils.StopWatch(duration=log_after)
            log_timer.start()
        while condition():
            if log_timer is not None and log_timer.expired():
                LOG.warning('Possible hang: %s', msg)
                LOG.debug(''.join(traceback.format_stack()))
                log_timer = None
            if timeout_timer is not None and timeout_timer.expired():
                raise TaskTimeout(msg)
            timeouts = []
            if log_timer is not None:
                timeouts.append(log_timer.leftover())
            if timeout_timer is not None:
                timeouts.append(timeout_timer.leftover())
            wait = None
            if timeouts:
                wait = min(timeouts)
            self._cond.wait(wait)

    @property
    def complete(self):
        return self._state == self.COMPLETE

    def wait_for_completion(self, caller, log_after, timeout_timer):
        """Wait until this task has completed.

        :param caller: The name of the task which is waiting.
        :param log_after: Emit a log message if waiting longer than `log_after`
                          seconds.
        :param timeout_timer: Raise TaskTimeout if StopWatch object
                              `timeout_timer` expires while waiting.
        """
        with self._cond:
            msg = '%s is waiting for %s to complete' % (caller, self._name)
            self._wait(lambda: not self.complete, msg, log_after, timeout_timer)

    def run_once(self, fn, log_after, timeout_timer):
        """Run a task exactly once. If it is currently running in another
        thread, wait for it to complete. If it has already run, return
        immediately without running it again.

        :param fn: The task to run. It must be a callable taking no arguments.
                   It may optionally return another callable, which also takes
                   no arguments, which will be executed after completion has
                   been signaled to other threads.
        :param log_after: Emit a log message if waiting longer than `log_after`
                          seconds.
        :param timeout_timer: Raise TaskTimeout if StopWatch object
                              `timeout_timer` expires while waiting.
        """
        with self._cond:
            if self._state == self.INIT:
                self._state = self.RUNNING
                self._cond.release()
                try:
                    post_fn = fn()
                finally:
                    self._cond.acquire()
                    self._state = self.COMPLETE
                    self._cond.notify_all()
                if post_fn is not None:
                    self._cond.release()
                    try:
                        post_fn()
                    finally:
                        self._cond.acquire()
            elif self._state == self.RUNNING:
                msg = '%s is waiting for another thread to complete' % self._name
                self._wait(lambda: self._state == self.RUNNING, msg, log_after, timeout_timer)