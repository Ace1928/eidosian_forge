import struct
from . import bpdu
from . import packet_base
from os_ken.lib import stringify
LLC sub encoder/decoder class for control U-format field.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte
    order.
    __init__ takes the corresponding args in this order.

    ======================== ===============================
    Attribute                Description
    ======================== ===============================
    modifier_function1       modifier function bit
    pf_bit                   poll/final bit
    modifier_function2       modifier function bit
    ======================== ===============================
    