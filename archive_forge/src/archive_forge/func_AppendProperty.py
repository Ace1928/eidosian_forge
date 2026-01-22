import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def AppendProperty(self, key, value):
    if key not in self._schema:
        raise KeyError(key + ' not in ' + self.__class__.__name__)
    is_list, property_type, is_strong = self._schema[key][0:3]
    if not is_list:
        raise TypeError(key + ' of ' + self.__class__.__name__ + ' must be list')
    if not isinstance(value, property_type):
        raise TypeError('item of ' + key + ' of ' + self.__class__.__name__ + ' must be ' + property_type.__name__ + ', not ' + value.__class__.__name__)
    self._properties[key] = self._properties.get(key, [])
    if is_strong:
        value.parent = self
    self._properties[key].append(value)