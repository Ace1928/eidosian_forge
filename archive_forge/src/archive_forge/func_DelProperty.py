import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def DelProperty(self, key):
    if key in self._properties:
        del self._properties[key]