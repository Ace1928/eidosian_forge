from __future__ import unicode_literals
import struct
from six import int2byte, binary_type, iterbytes
from .log import logger
def wont_received(self, data):
    """ Received telnet WONT command. """
    logger.info('WONT %r', data)