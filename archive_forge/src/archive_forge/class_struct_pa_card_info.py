import ctypes
from ctypes import *
import pyglet.lib
class struct_pa_card_info(Structure):
    __slots__ = ['index', 'name', 'owner_module', 'driver', 'n_profiles', 'profiles', 'active_profile', 'proplist', 'n_ports', 'ports', 'profiles2', 'active_profile2']