import ctypes
from ctypes import *
import pyglet.lib
class struct_pa_sink_input_info(Structure):
    __slots__ = ['index', 'name', 'owner_module', 'client', 'sink', 'sample_spec', 'channel_map', 'volume', 'buffer_usec', 'sink_usec', 'resample_method', 'driver', 'mute', 'proplist', 'corked', 'has_volume', 'volume_writable', 'format']