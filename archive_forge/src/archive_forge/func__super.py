from __future__ import annotations
import codecs
import contextlib
import sys
import typing
import warnings
from contextlib import suppress
from urwid import str_util
@property
def _super(self):
    warnings.warn(f'`{name}.__super` was a deprecated feature for old python versions.Please use `super()` call instead.', DeprecationWarning, stacklevel=3)
    return super(cls, self)