from __future__ import absolute_import
import platform
from ctypes import (
from ctypes.util import find_library
from ...packages.six import raise_from
class CFConst(object):
    """
    A class object that acts as essentially a namespace for CoreFoundation
    constants.
    """
    kCFStringEncodingUTF8 = CFStringEncoding(134217984)