import ctypes
from ctypes import *
import pyglet.lib
class struct_pa_card_profile_info2(Structure):
    __slots__ = ['name', 'description', 'n_sinks', 'n_sources', 'priority', 'available']