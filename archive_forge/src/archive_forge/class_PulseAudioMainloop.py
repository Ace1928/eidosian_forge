import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
class PulseAudioMainloop:

    def __init__(self) -> None:
        self._pa_threaded_mainloop = pa.pa_threaded_mainloop_new()
        self._pa_mainloop_vtab = pa.pa_threaded_mainloop_get_api(self._pa_threaded_mainloop)
        self.lock = _MainloopLock(self)

    def start(self) -> None:
        """Start running the mainloop."""
        result = pa.pa_threaded_mainloop_start(self._pa_threaded_mainloop)
        if result < 0:
            raise PulseAudioException(0, 'Failed to start PulseAudio mainloop')
        assert _debug('PulseAudioMainloop: Started')

    def delete(self) -> None:
        """Clean up the mainloop."""
        if self._pa_threaded_mainloop is not None:
            assert _debug('Delete PulseAudioMainloop')
            pa.pa_threaded_mainloop_stop(self._pa_threaded_mainloop)
            pa.pa_threaded_mainloop_free(self._pa_threaded_mainloop)
            self._pa_threaded_mainloop = None
            self._pa_mainloop_vtab = None

    def lock_(self) -> None:
        """Lock the threaded mainloop against events.  Required for all
        calls into PA."""
        assert self._pa_threaded_mainloop is not None
        pa.pa_threaded_mainloop_lock(self._pa_threaded_mainloop)

    def unlock(self) -> None:
        """Unlock the mainloop thread."""
        assert self._pa_threaded_mainloop is not None
        pa.pa_threaded_mainloop_unlock(self._pa_threaded_mainloop)

    def signal(self) -> None:
        """Signal the mainloop thread to break from a wait."""
        assert self._pa_threaded_mainloop is not None
        pa.pa_threaded_mainloop_signal(self._pa_threaded_mainloop, 0)

    def wait(self) -> None:
        """Unlock and then Wait for a signal from the locked mainloop.
        It's important to note that the PA mainloop lock is reentrant, yet this method only
        releases one lock.
        Before returning, the lock is reacquired.
        """
        assert self._pa_threaded_mainloop is not None
        pa.pa_threaded_mainloop_wait(self._pa_threaded_mainloop)

    def create_context(self) -> 'PulseAudioContext':
        """Construct and return a new context in this mainloop.
        Will grab the lock.
        """
        assert self._pa_mainloop_vtab is not None
        app_name = self._get_app_name().encode('utf-8')
        with self.lock:
            return PulseAudioContext(self, app_name)

    def _get_app_name(self) -> str:
        """Get the application name as advertised to the pulseaudio server."""
        return sys.argv[0]