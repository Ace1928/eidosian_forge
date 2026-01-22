import ctypes
from ctypes import *
import pyglet.lib
class struct_pa_card_port_info(Structure):
    __slots__ = ['name', 'description', 'priority', 'available', 'direction', 'n_profiles', 'profiles', 'proplist', 'latency_offset', 'profiles2']