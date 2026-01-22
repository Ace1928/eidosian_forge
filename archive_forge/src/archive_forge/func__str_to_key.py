import array
import logging
import posixpath
import warnings
from collections.abc import MutableMapping
from functools import cached_property
from fsspec.core import url_to_fs
def _str_to_key(self, s):
    """Strip path of to leave key name"""
    return s[len(self.root):].lstrip('/')