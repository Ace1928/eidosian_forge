import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
def _moved_callback(self, _stream, _userdata) -> None:
    self._refresh_sink_index()
    assert _debug(f'PulseAudioStream: moved to new index {self.index}')