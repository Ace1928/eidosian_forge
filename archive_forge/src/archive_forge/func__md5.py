import base64
import hashlib
import mmap
import os
import sys
from pathlib import Path
from typing import NewType, Union
from wandb.sdk.lib.paths import StrPath
def _md5(data: bytes=b'') -> 'hashlib._Hash':
    """Allow FIPS-compliant md5 hash when supported."""
    if sys.version_info >= (3, 9):
        return hashlib.md5(data, usedforsecurity=False)
    else:
        return hashlib.md5(data)