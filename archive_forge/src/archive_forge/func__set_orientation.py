from collections import deque
import math
import threading
from typing import Deque, Tuple, TYPE_CHECKING
from pyglet.media.drivers.base import AbstractAudioDriver, AbstractAudioPlayer, MediaEvent
from pyglet.media.player_worker_thread import PlayerWorkerThread
from pyglet.media.drivers.listener import AbstractListener
from pyglet.util import debug_print
from . import interface
def _set_orientation(self) -> None:
    self._xa2_listener.orientation = _convert_coordinates(self._forward_orientation) + _convert_coordinates(self._up_orientation)