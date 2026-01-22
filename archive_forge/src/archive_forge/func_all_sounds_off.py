from ctypes import *
from ctypes.util import find_library
import os
def all_sounds_off(self, chan):
    """Turn off all sounds on a channel (equivalent to mute)"""
    return fluid_synth_all_sounds_off(self.synth, chan)