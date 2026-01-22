from io import StringIO, BytesIO
import codecs
import os
import sys
import re
import errno
from .exceptions import ExceptionPexpect, EOF, TIMEOUT
from .expect import Expecter, searcher_string, searcher_re
def _set_buffer(self, value):
    self._buffer = self.buffer_type()
    self._buffer.write(value)