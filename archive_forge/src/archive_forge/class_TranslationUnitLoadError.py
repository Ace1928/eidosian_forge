from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class TranslationUnitLoadError(Exception):
    """Represents an error that occurred when loading a TranslationUnit.

    This is raised in the case where a TranslationUnit could not be
    instantiated due to failure in the libclang library.

    FIXME: Make libclang expose additional error information in this scenario.
    """
    pass