from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class nvmlClkMonFaultInfo_t(Structure):
    _fields_ = [('clkApiDomain', c_uint), ('clkDomainFaultMask', c_uint)]