from __future__ import absolute_import
import logging
import socket
import struct
import threading
import time
import serial
from serial.serialutil import SerialBase, SerialException, to_bytes, \
def _telnet_process_command(self, command):
    """Process commands other than DO, DONT, WILL, WONT."""
    if self.logger:
        self.logger.warning('ignoring Telnet command: {!r}'.format(command))