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
def _get_package_data_output_mapping(self) -> Iterator[Tuple[str, str]]:
    """Iterate over package data producing (dest, src) pairs."""
    for package, src_dir, build_dir, filenames in self.data_files:
        for filename in filenames:
            target = os.path.join(build_dir, filename)
            srcfile = os.path.join(src_dir, filename)
            yield (target, srcfile)