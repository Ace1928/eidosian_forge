from __future__ import absolute_import
import sys
import time
import serial
from serial.serialutil import  to_bytes
def tx(self, data):
    """show transmitted data as hex dump"""
    if self.color:
        self.output.write(self.tx_color)
    for offset, row in hexdump(data):
        self.write_line(time.time() - self.start_time, 'TX', '{:04X}  '.format(offset), row)