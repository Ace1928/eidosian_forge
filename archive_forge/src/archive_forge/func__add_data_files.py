from distutils import log
import distutils.command.sdist as orig
import os
import sys
import io
import contextlib
from itertools import chain
from .._importlib import metadata
from .build import _ORIGINAL_SUBCOMMANDS
def _add_data_files(self, data_files):
    """
        Add data files as found in build_py.data_files.
        """
    self.filelist.extend((os.path.join(src_dir, name) for _, src_dir, _, filenames in data_files for name in filenames))