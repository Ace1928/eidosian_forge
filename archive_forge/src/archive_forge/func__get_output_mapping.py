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
def _get_output_mapping(self) -> Iterator[Tuple[str, str]]:
    if not self.inplace:
        return
    build_py = self.get_finalized_command('build_py')
    opt = self.get_finalized_command('install_lib').optimize or ''
    for ext in self.extensions:
        inplace_file, regular_file = self._get_inplace_equivalent(build_py, ext)
        yield (regular_file, inplace_file)
        if ext._needs_stub:
            inplace_stub = self._get_equivalent_stub(ext, inplace_file)
            regular_stub = self._get_equivalent_stub(ext, regular_file)
            inplace_cache = _compiled_file_name(inplace_stub, optimization=opt)
            output_cache = _compiled_file_name(regular_stub, optimization=opt)
            yield (output_cache, inplace_cache)