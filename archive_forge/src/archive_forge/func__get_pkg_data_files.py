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
def _get_pkg_data_files(self, package):
    src_dir = self.get_package_dir(package)
    build_dir = os.path.join(*[self.build_lib] + package.split('.'))
    filenames = [os.path.relpath(file, src_dir) for file in self.find_data_files(package, src_dir)]
    return (package, src_dir, build_dir, filenames)