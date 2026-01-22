from __future__ import print_function, unicode_literals
import typing
from typing import Iterable
import six
from ._typing import Text
class _PermProperty(object):
    """Creates simple properties to get/set permissions."""

    def __init__(self, name):
        self._name = name
        self.__doc__ = "Boolean for '{}' permission.".format(name)

    def __get__(self, obj, obj_type=None):
        return self._name in obj

    def __set__(self, obj, value):
        if value:
            obj.add(self._name)
        else:
            obj.remove(self._name)