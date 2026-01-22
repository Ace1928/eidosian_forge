import logging
import os.path
import sys
from .exceptions import NoSuchClassError, UnsupportedPropertyError
from .icon_cache import IconCache
def _iconset(self, prop):
    return self.icon_cache.get_icon(prop)