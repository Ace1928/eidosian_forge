from collections import deque
from typing import TYPE_CHECKING, List, Optional, Tuple
import weakref
from pyglet.media.drivers.base import AbstractAudioDriver, AbstractAudioPlayer, MediaEvent
from pyglet.media.drivers.listener import AbstractListener
from pyglet.media.drivers.openal import interface
from pyglet.media.player_worker_thread import PlayerWorkerThread
from pyglet.util import debug_print
def _maybe_refill(self) -> bool:
    if self._pyglet_source_exhausted:
        return False
    remaining_bytes = self._write_cursor - self._play_cursor
    if remaining_bytes >= self._buffered_data_comfortable_limit:
        return False
    missing_bytes = self._buffered_data_ideal_size - remaining_bytes
    self._refill(self.source.audio_format.align_ceil(missing_bytes))
    return True