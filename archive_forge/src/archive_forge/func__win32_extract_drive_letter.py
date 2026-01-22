import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
def _win32_extract_drive_letter(url_base, path):
    """On win32 the drive letter needs to be added to the url base."""
    if len(path) < 4 or path[2] not in ':|' or path[3] != '/':
        raise InvalidURL(url_base + path, 'win32 file:/// paths need a drive letter')
    url_base += path[0:3]
    path = path[3:]
    return (url_base, path)