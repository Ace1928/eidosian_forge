from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class TranslationUnitSaveError(Exception):
    """Represents an error that occurred when saving a TranslationUnit.

    Each error has associated with it an enumerated value, accessible under
    e.save_error. Consumers can compare the value with one of the ERROR_
    constants in this class.
    """
    ERROR_UNKNOWN = 1
    ERROR_TRANSLATION_ERRORS = 2
    ERROR_INVALID_TU = 3

    def __init__(self, enumeration, message):
        assert isinstance(enumeration, int)
        if enumeration < 1 or enumeration > 3:
            raise Exception('Encountered undefined TranslationUnit save error constant: %d. Please file a bug to have this value supported.' % enumeration)
        self.save_error = enumeration
        Exception.__init__(self, 'Error %d: %s' % (enumeration, message))