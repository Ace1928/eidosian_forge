from __future__ import absolute_import
import errno
import logging
import select
import socket
import time
from serial.serialutil import SerialBase, SerialException, to_bytes, \
def _update_rts_state(self):
    """Set terminal status line: Request To Send"""
    if self.logger:
        self.logger.info('ignored _update_rts_state({!r})'.format(self._rts_state))