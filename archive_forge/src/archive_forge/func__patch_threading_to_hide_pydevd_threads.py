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
def _patch_threading_to_hide_pydevd_threads():
    """
    Patches the needed functions on the `threading` module so that the pydevd threads are hidden.

    Note that we patch the functions __code__ to avoid issues if some code had already imported those
    variables prior to the patching.
    """
    found_load_names = _collect_load_names(threading.enumerate)
    new_threading_enumerate = None
    if found_load_names in ({'_active_limbo_lock', '_limbo', '_active', 'values', 'list'}, {'_active_limbo_lock', '_limbo', '_active', 'values', 'NULL + list'}):
        pydev_log.debug('Applying patching to hide pydevd threads (Py3 version).')

        def new_threading_enumerate():
            with _active_limbo_lock:
                ret = list(_active.values()) + list(_limbo.values())
            return [t for t in ret if not getattr(t, 'is_pydev_daemon_thread', False)]
    elif found_load_names == set(('_active_limbo_lock', '_limbo', '_active', 'values')):
        pydev_log.debug('Applying patching to hide pydevd threads (Py2 version).')

        def new_threading_enumerate():
            with _active_limbo_lock:
                ret = _active.values() + _limbo.values()
            return [t for t in ret if not getattr(t, 'is_pydev_daemon_thread', False)]
    else:
        pydev_log.info('Unable to hide pydevd threads. Found names in threading.enumerate: %s', found_load_names)
    if new_threading_enumerate is not None:

        def pydevd_saved_threading_enumerate():
            with threading._active_limbo_lock:
                return list(threading._active.values()) + list(threading._limbo.values())
        _pydev_saved_modules.pydevd_saved_threading_enumerate = pydevd_saved_threading_enumerate
        threading.enumerate.__code__ = new_threading_enumerate.__code__

        def new_active_count():
            return len(enumerate())
        threading.active_count.__code__ = new_active_count.__code__
        if hasattr(threading, '_pickSomeNonDaemonThread'):

            def new_pick_some_non_daemon_thread():
                with _active_limbo_lock:
                    threads = list(_active.values()) + list(_limbo.values())
                for t in threads:
                    if not t.daemon and t.is_alive():
                        return t
                return None
            threading._pickSomeNonDaemonThread.__code__ = new_pick_some_non_daemon_thread.__code__