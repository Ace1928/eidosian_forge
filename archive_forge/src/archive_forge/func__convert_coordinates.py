from collections import deque
import math
import threading
from typing import Deque, Tuple, TYPE_CHECKING
from pyglet.media.drivers.base import AbstractAudioDriver, AbstractAudioPlayer, MediaEvent
from pyglet.media.player_worker_thread import PlayerWorkerThread
from pyglet.media.drivers.listener import AbstractListener
from pyglet.util import debug_print
from . import interface
def _convert_coordinates(coordinates: Tuple[float, float, float]) -> Tuple[float, float, float]:
    x, y, z = coordinates
    return (x, y, -z)