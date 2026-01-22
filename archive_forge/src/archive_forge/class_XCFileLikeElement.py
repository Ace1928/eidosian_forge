import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
class XCFileLikeElement(XCHierarchicalElement):

    def PathHashables(self):
        hashables = []
        xche = self
        while isinstance(xche, XCHierarchicalElement):
            xche_hashables = xche.Hashables()
            for index, xche_hashable in enumerate(xche_hashables):
                hashables.insert(index, xche_hashable)
            xche = xche.parent
        return hashables