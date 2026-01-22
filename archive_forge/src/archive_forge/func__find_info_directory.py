import io
import os
import shlex
import sys
import tokenize
import shutil
import contextlib
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union
import setuptools
import distutils
from . import errors
from ._path import same_path
from ._reqs import parse_strings
from .warnings import SetuptoolsDeprecationWarning
from distutils.util import strtobool
def _find_info_directory(self, metadata_directory: str, suffix: str) -> Path:
    for parent, dirs, _ in os.walk(metadata_directory):
        candidates = [f for f in dirs if f.endswith(suffix)]
        if len(candidates) != 0 or len(dirs) != 1:
            assert len(candidates) == 1, f'Multiple {suffix} directories found'
            return Path(parent, candidates[0])
    msg = f'No {suffix} directory found in {metadata_directory}'
    raise errors.InternalError(msg)