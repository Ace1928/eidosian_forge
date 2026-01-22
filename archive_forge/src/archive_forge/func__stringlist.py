import logging
import os.path
import sys
from .exceptions import NoSuchClassError, UnsupportedPropertyError
from .icon_cache import IconCache
def _stringlist(self, prop):
    return [self._string(p, notr='true') for p in prop]