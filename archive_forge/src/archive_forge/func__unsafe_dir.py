import errno
import fnmatch
import marshal
import os
import pickle
import stat
import sys
import tempfile
import typing as t
from hashlib import sha1
from io import BytesIO
from types import CodeType
def _unsafe_dir() -> 'te.NoReturn':
    raise RuntimeError('Cannot determine safe temp directory.  You need to explicitly provide one.')