import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
def _refresh_sink_index(self) -> None:
    self.index = pa.pa_stream_get_index(self._pa_stream)
    if self.index == PA_INVALID_INDEX:
        self.context().raise_error()