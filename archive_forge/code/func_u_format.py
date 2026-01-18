import functools
import os
import sys
import re
import shutil
import types
from .encoding import DEFAULT_ENCODING
import platform
@_modify_str_or_docstring
def u_format(s):
    """"{u}'abc'" --> "u'abc'" (Python 2)
        
        Accepts a string or a function, so it can be used as a decorator."""
    return s.format(u='u')