import logging
import os
import struct
import time
from pyu2f import errors
from pyu2f import hid
def InternalReadFrame(self):
    frame = self.hid_device.Read()
    self.logger.debug('recv: ' + str(frame))
    return frame