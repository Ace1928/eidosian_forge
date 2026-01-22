from collections import deque
import ctypes
import threading
from typing import Deque, Optional, TYPE_CHECKING
import weakref
from pyglet.media.drivers.base import AbstractAudioDriver, AbstractAudioPlayer, MediaEvent
from pyglet.media.drivers.listener import AbstractListener
from pyglet.media.player_worker_thread import PlayerWorkerThread
from pyglet.util import debug_print
from . import lib_pulseaudio as pa
from .interface import PulseAudioMainloop
def _get_read_index(self) -> int:
    if (t_info := self._latest_timing_info) is None:
        return 0
    read_idx = t_info.read_index - self._last_clear_read_index
    assert _debug(f'_get_read_index -> {read_idx}')
    return read_idx