import sys
from types import CodeType
import six
def wrapped_bytes(bstr):
    assert bstr.startswith('b')
    return bstr