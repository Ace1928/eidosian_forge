import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
def _win32_strip_local_trailing_slash(url):
    """Strip slashes after the drive letter"""
    if len(url) > WIN32_MIN_ABS_FILEURL_LENGTH:
        return url[:-1]
    else:
        return url