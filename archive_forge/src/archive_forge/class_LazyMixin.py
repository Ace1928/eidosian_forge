import binascii
import os
import mmap
import sys
import time
import errno
from io import BytesIO
from smmap import (
import hashlib
from gitdb.const import (
class LazyMixin:
    """
    Base class providing an interface to lazily retrieve attribute values upon
    first access. If slots are used, memory will only be reserved once the attribute
    is actually accessed and retrieved the first time. All future accesses will
    return the cached value as stored in the Instance's dict or slot.
    """
    __slots__ = tuple()

    def __getattr__(self, attr):
        """
        Whenever an attribute is requested that we do not know, we allow it
        to be created and set. Next time the same attribute is requested, it is simply
        returned from our dict/slots. """
        self._set_cache_(attr)
        return object.__getattribute__(self, attr)

    def _set_cache_(self, attr):
        """
        This method should be overridden in the derived class.
        It should check whether the attribute named by attr can be created
        and cached. Do nothing if you do not know the attribute or call your subclass

        The derived class may create as many additional attributes as it deems
        necessary in case a git command returns more information than represented
        in the single attribute."""
        pass