import functools
import itertools
import logging
import os
import posixpath
import re
import urllib.parse
from dataclasses import dataclass
from typing import (
from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.filetypes import WHEEL_EXTENSION
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.misc import (
from pip._internal.utils.models import KeyBasedCompareMixin
from pip._internal.utils.urls import path_to_url, url_to_path
def _clean_url_path(path: str, is_local_path: bool) -> str:
    """
    Clean the path portion of a URL.
    """
    if is_local_path:
        clean_func = _clean_file_url_path
    else:
        clean_func = _clean_url_path_part
    parts = _reserved_chars_re.split(path)
    cleaned_parts = []
    for to_clean, reserved in pairwise(itertools.chain(parts, [''])):
        cleaned_parts.append(clean_func(to_clean))
        cleaned_parts.append(reserved.upper())
    return ''.join(cleaned_parts)