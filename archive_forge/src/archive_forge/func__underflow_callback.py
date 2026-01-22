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
def _underflow_callback(self, _stream, _userdata) -> None:
    assert _debug('PulseAudioPlayer: underflow')
    with self._audio_data_lock:
        if self._pyglet_source_exhausted and self._audio_data_buffer.available == 0:
            MediaEvent('on_eos').sync_dispatch_to_player(self.player)
        self._has_underrun = True
    self.stream.mainloop.signal()