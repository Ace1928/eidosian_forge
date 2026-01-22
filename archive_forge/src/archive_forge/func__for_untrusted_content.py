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
def _for_untrusted_content(self, ctype, why):
    return ContentDisallowed('Refusing to deserialize {} content of type {}'.format(why, parenthesize_alias(self.type_to_name.get(ctype, ctype), ctype)))