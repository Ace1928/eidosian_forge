from __future__ import absolute_import, division, print_function
from base64 import (
from binascii import b2a_base64, a2b_base64, Error as _BinAsciiError
import logging
from passlib import exc
from passlib.utils.compat import (
from passlib.utils.decor import memoized_property
class LazyBase64Engine(Base64Engine):
    """Base64Engine which delays initialization until it's accessed"""
    _lazy_opts = None

    def __init__(self, *args, **kwds):
        self._lazy_opts = (args, kwds)

    def _lazy_init(self):
        args, kwds = self._lazy_opts
        super(LazyBase64Engine, self).__init__(*args, **kwds)
        del self._lazy_opts
        self.__class__ = Base64Engine

    def __getattribute__(self, attr):
        if not attr.startswith('_'):
            self._lazy_init()
        return object.__getattribute__(self, attr)