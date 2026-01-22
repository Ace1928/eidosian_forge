import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def EnsureNoIDCollisions(self):
    """Verifies that no two objects have the same ID.  Checks all descendants.
    """
    ids = {}
    descendants = self.Descendants()
    for descendant in descendants:
        if descendant.id in ids:
            other = ids[descendant.id]
            raise KeyError('Duplicate ID %s, objects "%s" and "%s" in "%s"' % (descendant.id, str(descendant._properties), str(other._properties), self._properties['rootObject'].Name()))
        ids[descendant.id] = descendant