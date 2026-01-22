from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class LinkageKind(BaseEnumeration):
    """Describes the kind of linkage of a cursor."""
    _kinds = []
    _name_map = None

    def from_param(self):
        return self.value

    def __repr__(self):
        return 'LinkageKind.%s' % (self.name,)