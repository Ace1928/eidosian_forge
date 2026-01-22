import logging
import os.path
import sys
from .exceptions import NoSuchClassError, UnsupportedPropertyError
from .icon_cache import IconCache
def _rect(self, prop):
    return QtCore.QRect(*int_list(prop))