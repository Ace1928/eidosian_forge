import io
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import (
from mypy_extensions import mypyc_attr
from packaging.specifiers import InvalidSpecifier, Specifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from pathspec import PathSpec
from pathspec.patterns.gitwildmatch import GitWildMatchPatternError
from black.handle_ipynb_magics import jupyter_dependencies_are_installed
from black.mode import TargetVersion
from black.output import err
from black.report import Report
def _path_is_ignored(root_relative_path: str, root: Path, gitignore_dict: Dict[Path, PathSpec]) -> bool:
    path = root / root_relative_path
    for gitignore_path, pattern in gitignore_dict.items():
        try:
            relative_path = path.relative_to(gitignore_path).as_posix()
        except ValueError:
            break
        if pattern.match_file(relative_path):
            return True
    return False