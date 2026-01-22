from __future__ import annotations
import io
import logging
import os
import threading
import warnings
import weakref
from errno import ESPIPE
from glob import has_magic
from hashlib import sha256
from typing import ClassVar
from .callbacks import DEFAULT_CALLBACK
from .config import apply_config, conf
from .dircache import DirCache
from .transaction import Transaction
from .utils import (
def _ls_from_cache(self, path):
    """Check cache for listing

        Returns listing, if found (may be empty list for a directly that exists
        but contains nothing), None if not in cache.
        """
    parent = self._parent(path)
    if path.rstrip('/') in self.dircache:
        return self.dircache[path.rstrip('/')]
    try:
        files = [f for f in self.dircache[parent] if f['name'] == path or (f['name'] == path.rstrip('/') and f['type'] == 'directory')]
        if len(files) == 0:
            raise FileNotFoundError(path)
        return files
    except KeyError:
        pass