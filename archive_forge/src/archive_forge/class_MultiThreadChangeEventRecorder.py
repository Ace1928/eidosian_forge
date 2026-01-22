import inspect
import os
import threading
from contextlib import contextmanager
from datetime import datetime
from traits import trait_notifiers
class MultiThreadChangeEventRecorder(object):
    """ A thread aware trait change recorder.

    The class manages multiple ChangeEventRecorders which record trait change
    events for each thread in a separate file.

    Parameters
    ----------
    container : MultiThreadChangeEventRecorder
        The container of RecordContainers to keep the trait change records
        for each thread.

    Attributes
    ----------
    container : MultiThreadChangeEventRecorder
        The container of RecordContainers to keep the trait change records
        for each thread.
    tracers : dict
        Mapping from threads to ChangeEventRecorder instances.
    """

    def __init__(self, container):
        self.tracers = {}
        self._tracer_lock = threading.Lock()
        self.container = container

    def close(self):
        """ Close and stop all logging.

        """
        with self._tracer_lock:
            self.tracers = {}

    def pre_tracer(self, obj, name, old, new, handler):
        """ The traits pre event tracer.

        This method should be set as the global pre event tracer for traits.

        """
        tracer = self._get_tracer()
        tracer.pre_tracer(obj, name, old, new, handler)

    def post_tracer(self, obj, name, old, new, handler, exception=None):
        """ The traits post event tracer.

        This method should be set as the global post event tracer for traits.

        """
        tracer = self._get_tracer()
        tracer.post_tracer(obj, name, old, new, handler, exception=exception)

    def _get_tracer(self):
        with self._tracer_lock:
            thread = threading.current_thread().name
            if thread not in self.tracers:
                container = self.container
                thread_container = container.get_change_event_collector(thread)
                tracer = ChangeEventRecorder(thread_container)
                self.tracers[thread] = tracer
                return tracer
            else:
                return self.tracers[thread]