import logging
import os.path
import sys
from .exceptions import NoSuchClassError, UnsupportedPropertyError
from .icon_cache import IconCache
def _pyEnumMember(self, cpp_name):
    try:
        prefix, membername = cpp_name.split('::')
    except ValueError:
        prefix = 'Qt'
        membername = cpp_name
    if prefix == 'Qt':
        return getattr(QtCore.Qt, membername)
    scope = self.factory.findQObjectType(prefix)
    if scope is None:
        raise NoSuchClassError(prefix)
    return getattr(scope, membername)