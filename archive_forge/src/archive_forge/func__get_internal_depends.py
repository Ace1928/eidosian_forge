import os
import sys
import itertools
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.util import cache_from_source as _compiled_file_name
from typing import Dict, Iterator, List, Tuple
from pathlib import Path
from distutils.command.build_ext import build_ext as _du_build_ext
from distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler, get_config_var
from distutils import log
from setuptools.errors import BaseError
from setuptools.extension import Extension, Library
from distutils.sysconfig import _config_vars as _CONFIG_VARS  # noqa
def _get_internal_depends(self) -> Iterator[str]:
    """Yield ``ext.depends`` that are contained by the project directory"""
    project_root = Path(self.distribution.src_root or os.curdir).resolve()
    depends = (dep for ext in self.extensions for dep in ext.depends)

    def skip(orig_path: str, reason: str) -> None:
        log.info("dependency %s won't be automatically included in the manifest: the path %s", orig_path, reason)
    for dep in depends:
        path = Path(dep)
        if path.is_absolute():
            skip(dep, 'must be relative')
            continue
        if '..' in path.parts:
            skip(dep, "can't have `..` segments")
            continue
        try:
            resolved = (project_root / path).resolve(strict=True)
        except OSError:
            skip(dep, "doesn't exist")
            continue
        try:
            resolved.relative_to(project_root)
        except ValueError:
            skip(dep, 'must be inside the project root')
            continue
        yield path.as_posix()