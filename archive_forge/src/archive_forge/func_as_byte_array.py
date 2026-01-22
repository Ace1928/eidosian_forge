from __future__ import absolute_import
import System
import System.IO.Ports
from serial.serialutil import *
def as_byte_array(string):
    return sab([ord(x) for x in string])