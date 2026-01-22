from distutils import log
import distutils.command.sdist as orig
import os
import sys
import io
import contextlib
from itertools import chain
from .._importlib import metadata
from .build import _ORIGINAL_SUBCOMMANDS
def _default_to_gztar(self):
    if sys.version_info >= (3, 6, 0, 'beta', 1):
        return
    self.formats = ['gztar']