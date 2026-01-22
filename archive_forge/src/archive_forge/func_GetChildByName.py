import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def GetChildByName(self, name):
    if 'children' not in self._properties:
        return None
    for child in self._properties['children']:
        if child.Name() == name:
            return child
    return None