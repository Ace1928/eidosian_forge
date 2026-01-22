import os
import stat
import sys
import warnings
from contextlib import suppress
from io import BytesIO
from typing import (
from .errors import NotTreeError
from .file import GitFile
from .objects import (
from .pack import (
from .protocol import DEPTH_INFINITE
from .refs import PEELED_TAG_SUFFIX, Ref
def _want_deepen(sha):
    if not depth:
        return False
    if depth == DEPTH_INFINITE:
        return True
    return depth > self._get_depth(sha)