from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@property
def translation_unit(self):
    """The TranslationUnit to which this Type is associated."""
    return self._tu