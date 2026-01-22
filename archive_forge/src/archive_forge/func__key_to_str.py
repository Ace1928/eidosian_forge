import array
import logging
import posixpath
import warnings
from collections.abc import MutableMapping
from functools import cached_property
from fsspec.core import url_to_fs
def _key_to_str(self, key):
    """Generate full path for the key"""
    if not isinstance(key, str):
        warnings.warn('from fsspec 2023.5 onward FSMap non-str keys will raise TypeError', DeprecationWarning)
        if isinstance(key, list):
            key = tuple(key)
        key = str(key)
    return f'{self._root_key_to_str}{key}'