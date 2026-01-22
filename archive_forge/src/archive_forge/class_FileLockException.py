from __future__ import annotations
import bz2
import errno
import gzip
import io
import mmap
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING
class FileLockException(Exception):
    """Exception raised by FileLock."""