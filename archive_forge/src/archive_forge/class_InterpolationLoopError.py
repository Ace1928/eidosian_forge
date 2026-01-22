import os
import re
import sys
from codecs import BOM_UTF8, BOM_UTF16, BOM_UTF16_BE, BOM_UTF16_LE
import six
from ._version import __version__
class InterpolationLoopError(InterpolationError):
    """Maximum interpolation depth exceeded in string interpolation."""

    def __init__(self, option):
        InterpolationError.__init__(self, 'interpolation loop detected in value "%s".' % option)