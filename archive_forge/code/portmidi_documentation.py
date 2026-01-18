import ctypes
import threading
from ..ports import BaseInput, BaseOutput, sleep
from . import portmidi_init as pm
Stop callback thread if running.