from __future__ import annotations
import errno
import hashlib
import json
import logging
import os
import sys
from collections.abc import Callable, Hashable, Iterable
from pathlib import Path
from typing import (
import requests
from filelock import FileLock
def _key_to_cachefile_path(self, namespace: str, key: str | dict[str, Hashable]) -> str:
    namespace_path = str(Path(self.cache_dir, namespace))
    hashed_key = _make_cache_key(key)
    cache_path = str(Path(namespace_path, hashed_key + self.file_ext))
    return cache_path