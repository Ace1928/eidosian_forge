from __future__ import annotations
import os
import sys
from contextlib import suppress
from errno import ENOSYS
from typing import cast
from ._api import BaseFileLock
from ._util import ensure_directory_exists
Uses the :func:`fcntl.flock` to hard lock the lock file on unix systems.