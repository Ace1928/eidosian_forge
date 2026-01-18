from __future__ import unicode_literals
import struct
from six import int2byte, binary_type, iterbytes
from .log import logger
def will_received(self, data):
    """ Received telnet WILL command. """
    logger.info('WILL %r', data)