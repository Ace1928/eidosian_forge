from __future__ import absolute_import
import logging
import numbers
import time
from serial.serialutil import SerialBase, SerialException, to_bytes, iterbytes, SerialTimeoutException, PortNotOpenError
Read terminal status line: Carrier Detect