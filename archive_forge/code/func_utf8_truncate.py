from passlib.utils.compat import JYTHON
from binascii import b2a_base64, a2b_base64, Error as _BinAsciiError
from base64 import b64encode, b64decode
from codecs import lookup as _lookup_codec
from functools import update_wrapper
import itertools
import inspect
import logging; log = logging.getLogger(__name__)
import math
import os
import sys
import random
import re
import time
import timeit
import types
from warnings import warn
from passlib.utils.binary import (
from passlib.utils.decor import (
from passlib.exc import ExpectedStringError, ExpectedTypeError
from passlib.utils.compat import (add_doc, join_bytes, join_byte_values,
from passlib.exc import MissingBackendError
def utf8_truncate(source, index):
    """
    helper to truncate UTF8 byte string to nearest character boundary ON OR AFTER <index>.
    returned prefix will always have length of at least <index>, and will stop on the
    first byte that's not a UTF8 continuation byte (128 - 191 inclusive).
    since utf8 should never take more than 4 bytes to encode known unicode values,
    we can stop after ``index+3`` is reached.

    :param bytes source:
    :param int index:
    :rtype: bytes
    """
    if not isinstance(source, bytes):
        raise ExpectedTypeError(source, bytes, 'source')
    end = len(source)
    if index < 0:
        index = max(0, index + end)
    if index >= end:
        return source
    end = min(index + 3, end)
    while index < end:
        if byte_elem_value(source[index]) & 192 != 128:
            break
        index += 1
    else:
        assert index == end
    result = source[:index]

    def sanity_check():
        try:
            text = source.decode('utf-8')
        except UnicodeDecodeError:
            return True
        assert text.startswith(result.decode('utf-8'))
        return True
    assert sanity_check()
    return result