import base64
import ctypes
import itertools
import os
import re
import ssl
import struct
import tempfile
from .bindings import CFConst, CoreFoundation, Security
def _cf_dictionary_from_tuples(tuples):
    """
    Given a list of Python tuples, create an associated CFDictionary.
    """
    dictionary_size = len(tuples)
    keys = (t[0] for t in tuples)
    values = (t[1] for t in tuples)
    cf_keys = (CoreFoundation.CFTypeRef * dictionary_size)(*keys)
    cf_values = (CoreFoundation.CFTypeRef * dictionary_size)(*values)
    return CoreFoundation.CFDictionaryCreate(CoreFoundation.kCFAllocatorDefault, cf_keys, cf_values, dictionary_size, CoreFoundation.kCFTypeDictionaryKeyCallBacks, CoreFoundation.kCFTypeDictionaryValueCallBacks)