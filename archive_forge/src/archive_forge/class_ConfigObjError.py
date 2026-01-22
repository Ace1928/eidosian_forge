import os
import re
import sys
from codecs import BOM_UTF8, BOM_UTF16, BOM_UTF16_BE, BOM_UTF16_LE
import six
from ._version import __version__
class ConfigObjError(SyntaxError):
    """
    This is the base class for all errors that ConfigObj raises.
    It is a subclass of SyntaxError.
    """

    def __init__(self, message='', line_number=None, line=''):
        self.line = line
        self.line_number = line_number
        SyntaxError.__init__(self, message)