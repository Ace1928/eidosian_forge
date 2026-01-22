from __future__ import absolute_import
import ctypes
import time
from serial import win32
import serial
from serial.serialutil import SerialBase, SerialException, to_bytes, PortNotOpenError, SerialTimeoutException
def _GetCommModemStatus(self):
    if not self.is_open:
        raise PortNotOpenError()
    stat = win32.DWORD()
    win32.GetCommModemStatus(self._port_handle, ctypes.byref(stat))
    return stat.value