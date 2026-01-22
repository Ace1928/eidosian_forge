import base64
import ctypes
import itertools
import os
import re
import ssl
import struct
import tempfile
from .bindings import CFConst, CoreFoundation, Security
def _cf_data_from_bytes(bytestring):
    """
    Given a bytestring, create a CFData object from it. This CFData object must
    be CFReleased by the caller.
    """
    return CoreFoundation.CFDataCreate(CoreFoundation.kCFAllocatorDefault, bytestring, len(bytestring))