import six
from six.moves import builtins
from six.moves import range
import struct
import sys
import time
import os
import zlib
import io
def _unread(self, buf):
    self.extrasize = len(buf) + self.extrasize
    self.offset -= len(buf)