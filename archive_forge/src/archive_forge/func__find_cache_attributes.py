from __future__ import absolute_import
import cython
import os
import sys
import re
import io
import codecs
import glob
import shutil
import tempfile
from functools import wraps
from . import __version__ as cython_version
def _find_cache_attributes(obj):
    """The function iterates over the attributes of the object and,
    if it finds the name of the cache, it returns it and the corresponding method name.
    The method may not be present in the object.
    """
    for attr_name in dir(obj):
        match = _CACHE_NAME_PATTERN.match(attr_name)
        if match is not None:
            yield (attr_name, match.group(1))