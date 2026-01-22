import codecs
import functools
import os
import pickle
import re
import sys
import textwrap
import zipfile
from abc import ABCMeta, abstractmethod
from gzip import WRITE as GZ_WRITE
from gzip import GzipFile
from io import BytesIO, TextIOWrapper
from urllib.request import url2pathname, urlopen
from nltk import grammar, sem
from nltk.compat import add_py3_data, py3_data
from nltk.internals import deprecated
def _incr_decode(self, bytes):
    """
        Decode the given byte string into a unicode string, using this
        reader's encoding.  If an exception is encountered that
        appears to be caused by a truncation error, then just decode
        the byte string without the bytes that cause the trunctaion
        error.

        Return a tuple ``(chars, num_consumed)``, where ``chars`` is
        the decoded unicode string, and ``num_consumed`` is the
        number of bytes that were consumed.
        """
    while True:
        try:
            return self.decode(bytes, 'strict')
        except UnicodeDecodeError as exc:
            if exc.end == len(bytes):
                return self.decode(bytes[:exc.start], self.errors)
            elif self.errors == 'strict':
                raise
            else:
                return self.decode(bytes, self.errors)