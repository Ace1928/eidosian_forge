from collections import deque
import math
import threading
from typing import Deque, Tuple, TYPE_CHECKING
from pyglet.media.drivers.base import AbstractAudioDriver, AbstractAudioPlayer, MediaEvent
from pyglet.media.player_worker_thread import PlayerWorkerThread
from pyglet.media.drivers.listener import AbstractListener
from pyglet.util import debug_print
from . import interface
def _set_cone_angles(self) -> None:
    inner = min(self._cone_inner_angle, self._cone_outer_angle)
    outer = max(self._cone_inner_angle, self._cone_outer_angle)
    self._xa2_source_voice.set_cone_angles(math.radians(inner), math.radians(outer))