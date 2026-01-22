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
class _IncludePackageDataAbuse:
    """Inform users that package or module is included as 'data file'"""

    class _Warning(SetuptoolsDeprecationWarning):
        _SUMMARY = '\n        Package {importable!r} is absent from the `packages` configuration.\n        '
        _DETAILS = '\n        ############################\n        # Package would be ignored #\n        ############################\n        Python recognizes {importable!r} as an importable package[^1],\n        but it is absent from setuptools\' `packages` configuration.\n\n        This leads to an ambiguous overall configuration. If you want to distribute this\n        package, please make sure that {importable!r} is explicitly added\n        to the `packages` configuration field.\n\n        Alternatively, you can also rely on setuptools\' discovery methods\n        (for example by using `find_namespace_packages(...)`/`find_namespace:`\n        instead of `find_packages(...)`/`find:`).\n\n        You can read more about "package discovery" on setuptools documentation page:\n\n        - https://setuptools.pypa.io/en/latest/userguide/package_discovery.html\n\n        If you don\'t want {importable!r} to be distributed and are\n        already explicitly excluding {importable!r} via\n        `find_namespace_packages(...)/find_namespace` or `find_packages(...)/find`,\n        you can try to use `exclude_package_data`, or `include-package-data=False` in\n        combination with a more fine grained `package-data` configuration.\n\n        You can read more about "package data files" on setuptools documentation page:\n\n        - https://setuptools.pypa.io/en/latest/userguide/datafiles.html\n\n\n        [^1]: For Python, any directory (with suitable naming) can be imported,\n              even if it does not contain any `.py` files.\n              On the other hand, currently there is no concept of package data\n              directory, all directories are treated like packages.\n        '

    def __init__(self):
        self._already_warned = set()

    def is_module(self, file):
        return file.endswith('.py') and file[:-len('.py')].isidentifier()

    def importable_subpackage(self, parent, file):
        pkg = Path(file).parent
        parts = list(itertools.takewhile(str.isidentifier, pkg.parts))
        if parts:
            return '.'.join([parent, *parts])
        return None

    def warn(self, importable):
        if importable not in self._already_warned:
            self._Warning.emit(importable=importable)
            self._already_warned.add(importable)