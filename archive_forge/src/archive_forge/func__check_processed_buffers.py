from collections import deque
from typing import TYPE_CHECKING, List, Optional, Tuple
import weakref
from pyglet.media.drivers.base import AbstractAudioDriver, AbstractAudioPlayer, MediaEvent
from pyglet.media.drivers.listener import AbstractListener
from pyglet.media.drivers.openal import interface
from pyglet.media.player_worker_thread import PlayerWorkerThread
from pyglet.util import debug_print
def _check_processed_buffers(self) -> None:
    buffers_processed = self.alsource.unqueue_buffers()
    for _ in range(buffers_processed):
        self._buffer_cursor += self._queued_buffer_sizes.popleft()