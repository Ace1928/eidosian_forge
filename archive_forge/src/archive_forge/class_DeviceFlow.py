from abc import ABCMeta, abstractmethod
from enum import Enum, auto
from typing import Dict, Optional
from pyglet import event
class DeviceFlow(Enum):
    OUTPUT = auto()
    INPUT = auto()
    INPUT_OUTPUT = auto()