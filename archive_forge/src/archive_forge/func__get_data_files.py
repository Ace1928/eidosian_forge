from functools import partial
from glob import glob
from distutils.util import convert_path
import distutils.command.build_py as orig
import os
import fnmatch
import textwrap
import io
import distutils.errors
import itertools
import stat
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
from ..extern.more_itertools import unique_everseen
from ..warnings import SetuptoolsDeprecationWarning
def _get_data_files(self):
    """Generate list of '(package,src_dir,build_dir,filenames)' tuples"""
    self.analyze_manifest()
    return list(map(self._get_pkg_data_files, self.packages or ()))