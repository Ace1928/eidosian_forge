import sys
import re
import warnings
import io
import collections
import collections.abc
import contextlib
import weakref
from . import ElementPath
fromstring = XML
def _raiseerror(self, value):
    err = ParseError(value)
    err.code = value.code
    err.position = (value.lineno, value.offset)
    raise err