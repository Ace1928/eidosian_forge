import os
import shutil
from contextlib import contextmanager
from distutils import log
from distutils.core import Command
from pathlib import Path
from .. import _normalization

    This command is private and reserved for internal use of setuptools,
    users should rely on ``setuptools.build_meta`` APIs.
    