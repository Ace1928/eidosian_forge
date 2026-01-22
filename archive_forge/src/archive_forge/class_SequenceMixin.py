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
class SequenceMixin(object):
    """
    helper which lets result object act like a fixed-length sequence.
    subclass just needs to provide :meth:`_as_tuple()`.
    """

    def _as_tuple(self):
        raise NotImplementedError('implement in subclass')

    def __repr__(self):
        return repr(self._as_tuple())

    def __getitem__(self, idx):
        return self._as_tuple()[idx]

    def __iter__(self):
        return iter(self._as_tuple())

    def __len__(self):
        return len(self._as_tuple())

    def __eq__(self, other):
        return self._as_tuple() == other

    def __ne__(self, other):
        return not self.__eq__(other)