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
def _set_special_baudrate(self, baudrate):
    buf = array.array('i', [baudrate])
    fcntl.ioctl(self.fd, IOSSIOSPEED, buf, 1)