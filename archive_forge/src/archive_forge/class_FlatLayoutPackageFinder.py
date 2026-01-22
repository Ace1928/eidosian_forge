import itertools
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import (
import _distutils_hack.override  # noqa: F401
from distutils import log
from distutils.util import convert_path
class FlatLayoutPackageFinder(PEP420PackageFinder):
    _EXCLUDE = ('ci', 'bin', 'debian', 'doc', 'docs', 'documentation', 'manpages', 'news', 'newsfragments', 'changelog', 'test', 'tests', 'unit_test', 'unit_tests', 'example', 'examples', 'scripts', 'tools', 'util', 'utils', 'python', 'build', 'dist', 'venv', 'env', 'requirements', 'tasks', 'fabfile', 'site_scons', 'benchmark', 'benchmarks', 'exercise', 'exercises', 'htmlcov', '[._]*')
    DEFAULT_EXCLUDE = tuple(chain_iter(((p, f'{p}.*') for p in _EXCLUDE)))
    'Reserved package names'

    @staticmethod
    def _looks_like_package(_path: _Path, package_name: str) -> bool:
        names = package_name.split('.')
        root_pkg_is_valid = names[0].isidentifier() or names[0].endswith('-stubs')
        return root_pkg_is_valid and all((name.isidentifier() for name in names[1:]))