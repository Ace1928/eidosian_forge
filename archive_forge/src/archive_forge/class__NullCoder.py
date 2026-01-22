from io import StringIO, BytesIO
import codecs
import os
import sys
import re
import errno
from .exceptions import ExceptionPexpect, EOF, TIMEOUT
from .expect import Expecter, searcher_string, searcher_re
class _NullCoder(object):
    """Pass bytes through unchanged."""

    @staticmethod
    def encode(b, final=False):
        return b

    @staticmethod
    def decode(b, final=False):
        return b