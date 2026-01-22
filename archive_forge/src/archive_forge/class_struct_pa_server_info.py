import ctypes
from ctypes import *
import pyglet.lib
class struct_pa_server_info(Structure):
    __slots__ = ['user_name', 'host_name', 'server_version', 'server_name', 'sample_spec', 'default_sink_name', 'default_source_name', 'cookie', 'channel_map']