from __future__ import absolute_import
import errno
import logging
import select
import socket
import time
from serial.serialutil import SerialBase, SerialException, to_bytes, \
def _update_dtr_state(self):
    """Set terminal status line: Data Terminal Ready"""
    if self.logger:
        self.logger.info('ignored _update_dtr_state({!r})'.format(self._dtr_state))