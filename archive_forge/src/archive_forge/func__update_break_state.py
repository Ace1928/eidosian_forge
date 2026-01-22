from __future__ import absolute_import
import errno
import logging
import select
import socket
import time
from serial.serialutil import SerialBase, SerialException, to_bytes, \
def _update_break_state(self):
    """Set break: Controls TXD. When active, to transmitting is
        possible."""
    if self.logger:
        self.logger.info('ignored _update_break_state({!r})'.format(self._break_state))