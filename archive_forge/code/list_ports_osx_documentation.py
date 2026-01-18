from __future__ import absolute_import
import ctypes
from serial.tools import list_ports_common

    helper function to scan USB interfaces
    returns a list of SuitableSerialInterface objects with name and id attributes
    