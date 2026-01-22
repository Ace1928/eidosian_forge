from __future__ import annotations
import codecs
import os
import pickle
import sys
from collections import namedtuple
from contextlib import contextmanager
from io import BytesIO
from .exceptions import (ContentDisallowed, DecodeError, EncodeError,
from .utils.compat import entrypoints
from .utils.encoding import bytes_to_str, str_to_bytes
@contextmanager
def _reraise_errors(wrapper, include=(Exception,), exclude=(SerializerNotInstalled,)):
    try:
        yield
    except exclude:
        raise
    except include as exc:
        reraise(wrapper, wrapper(exc), sys.exc_info()[2])