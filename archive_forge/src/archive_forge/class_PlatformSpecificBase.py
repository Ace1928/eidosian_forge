from __future__ import absolute_import
import errno
import fcntl
import os
import select
import struct
import sys
import termios
import serial
from serial.serialutil import SerialBase, SerialException, to_bytes, \
class PlatformSpecificBase(object):
    BAUDRATE_CONSTANTS = {}

    def _set_special_baudrate(self, baudrate):
        raise NotImplementedError('non-standard baudrates are not supported on this platform')

    def _set_rs485_mode(self, rs485_settings):
        raise NotImplementedError('RS485 not supported on this platform')

    def set_low_latency_mode(self, low_latency_settings):
        raise NotImplementedError('Low latency not supported on this platform')

    def _update_break_state(self):
        """        Set break: Controls TXD. When active, no transmitting is possible.
        """
        if self._break_state:
            fcntl.ioctl(self.fd, TIOCSBRK)
        else:
            fcntl.ioctl(self.fd, TIOCCBRK)