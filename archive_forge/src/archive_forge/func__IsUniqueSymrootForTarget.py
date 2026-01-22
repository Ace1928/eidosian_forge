import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def _IsUniqueSymrootForTarget(self, symroot):
    uniquifier = ['$SRCROOT', '$(SRCROOT)']
    if any((x in symroot for x in uniquifier)):
        return True
    return False