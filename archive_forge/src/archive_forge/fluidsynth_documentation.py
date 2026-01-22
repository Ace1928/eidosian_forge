from ctypes import *
from ctypes.util import find_library
import os
Create new sequencer object to control and schedule timing of midi events

        Optional keyword arguments:
        time_scale: ticks per second, defaults to 1000
        use_system_timer: whether the sequencer should advance by itself
        