from ctypes import *
from ctypes.util import find_library
import os
class fluid_midi_router_t(Structure):
    _fields_ = [('synth', c_void_p), ('rules_mutex', c_void_p), ('rules', c_void_p * 6), ('free_rules', c_void_p), ('event_handler', c_void_p), ('event_handler_data', c_void_p), ('nr_midi_channels', c_int), ('cmd_rule', c_void_p), ('cmd_rule_type', POINTER(c_int))]