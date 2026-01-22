from _pydev_bundle._pydev_saved_modules import threading
from _pydev_bundle import _pydev_saved_modules
from _pydevd_bundle.pydevd_utils import notify_about_gevent_if_needed
import weakref
from _pydevd_bundle.pydevd_constants import IS_JYTHON, IS_IRONPYTHON, \
from _pydev_bundle.pydev_log import exception as pydev_log_exception
import sys
from _pydev_bundle import pydev_log
import pydevd_tracing
from _pydevd_bundle.pydevd_collect_bytecode_info import iter_instructions
class PyDBDaemonThread(threading.Thread):

    def __init__(self, py_db, target_and_args=None):
        """
        :param target_and_args:
            tuple(func, args, kwargs) if this should be a function and args to run.
            -- Note: use through run_as_pydevd_daemon_thread().
        """
        threading.Thread.__init__(self)
        notify_about_gevent_if_needed()
        self._py_db = weakref.ref(py_db)
        self._kill_received = False
        mark_as_pydevd_daemon_thread(self)
        self._target_and_args = target_and_args

    @property
    def py_db(self):
        return self._py_db()

    def run(self):
        created_pydb_daemon = self.py_db.created_pydb_daemon_threads
        created_pydb_daemon[self] = 1
        try:
            try:
                if IS_JYTHON and (not isinstance(threading.current_thread(), threading._MainThread)):
                    ss = JyCore.PySystemState()
                    JyCore.Py.setSystemState(ss)
                self._stop_trace()
                self._on_run()
            except:
                if sys is not None and pydev_log_exception is not None:
                    pydev_log_exception()
        finally:
            del created_pydb_daemon[self]

    def _on_run(self):
        if self._target_and_args is not None:
            target, args, kwargs = self._target_and_args
            target(*args, **kwargs)
        else:
            raise NotImplementedError('Should be reimplemented by: %s' % self.__class__)

    def do_kill_pydev_thread(self):
        if not self._kill_received:
            pydev_log.debug('%s received kill signal', self.name)
            self._kill_received = True

    def _stop_trace(self):
        if self.pydev_do_not_trace:
            pydevd_tracing.SetTrace(None)