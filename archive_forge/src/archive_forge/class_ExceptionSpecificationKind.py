from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class ExceptionSpecificationKind(BaseEnumeration):
    """
    An ExceptionSpecificationKind describes the kind of exception specification
    that a function has.
    """
    _kinds = []
    _name_map = None

    def __repr__(self):
        return 'ExceptionSpecificationKind.{}'.format(self.name)