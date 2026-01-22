import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
def _set_state_callback(self, py_callback: Optional[Callable[['PulseAudioContext', Any], Any]]) -> None:
    if py_callback is None:
        self._pa_state_change_callback = None
    else:
        self._pa_state_change_callback = pa.pa_context_notify_cb_t(py_callback)
    pa.pa_context_set_state_callback(self._pa_context, self._pa_state_change_callback, None)