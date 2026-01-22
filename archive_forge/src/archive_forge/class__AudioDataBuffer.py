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
class _AudioDataBuffer:

    def __init__(self, ideal_size: int, comfortable_limit: int) -> None:
        self.available = 0
        self.virtual_write_index = 0
        self._ideal_size = ideal_size
        self._comfortable_limit = comfortable_limit
        self._data: Deque['AudioData'] = deque()
        self._first_read_offset = 0

    def clear(self) -> None:
        self.available = 0
        self.virtual_write_index = 0
        self._data.clear()
        self._first_read_offset = 0

    def get_ideal_refill_size(self, virtual_required: int=0) -> int:
        virtual_available = self.available - virtual_required
        if virtual_available < self._comfortable_limit:
            return self._ideal_size - virtual_available
        return 0

    def add_data(self, d: 'AudioData') -> None:
        self._data.append(d)
        self.available += d.length
        self.virtual_write_index += d.length

    def memmove(self, target_pointer: int, num_bytes: int) -> int:
        bytes_written = 0
        bytes_remaining = num_bytes
        while bytes_remaining > 0 and self._data:
            cur_audio_data = self._data[0]
            cur_len = cur_audio_data.length - self._first_read_offset
            packet_used = cur_len <= bytes_remaining
            cur_write = min(bytes_remaining, cur_len)
            ctypes.memmove(target_pointer + bytes_written, cur_audio_data.pointer + self._first_read_offset, cur_write)
            bytes_written += cur_write
            bytes_remaining -= cur_write
            if packet_used:
                self._data.popleft()
                self._first_read_offset = 0
            else:
                self._first_read_offset += cur_write
        self.available -= bytes_written
        return bytes_written