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
def version_mismatch(*args, **kwargs):
    raise SerializerNotInstalled('msgpack requires msgpack-python >= 0.4.0')