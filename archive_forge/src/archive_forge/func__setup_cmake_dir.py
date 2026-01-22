from __future__ import annotations
from .base import ExternalDependency, DependencyException, DependencyTypeName
from ..mesonlib import is_windows, MesonException, PerMachine, stringlistify, extract_as_list
from ..cmake import CMakeExecutor, CMakeTraceParser, CMakeException, CMakeToolchain, CMakeExecScope, check_cmake_args, resolve_cmake_trace_targets, cmake_is_debug
from .. import mlog
import importlib.resources
from pathlib import Path
import functools
import re
import os
import shutil
import textwrap
import typing as T
def _setup_cmake_dir(self, cmake_file: str) -> Path:
    build_dir = self._get_build_dir()
    cmake_cache = build_dir / 'CMakeCache.txt'
    cmake_files = build_dir / 'CMakeFiles'
    if cmake_cache.exists():
        cmake_cache.unlink()
    shutil.rmtree(cmake_files.as_posix(), ignore_errors=True)
    cmake_txt = importlib.resources.read_text('mesonbuild.dependencies.data', cmake_file, encoding='utf-8')
    from ..cmake import language_map
    cmake_language = [language_map[x] for x in self.language_list if x in language_map]
    if not cmake_language:
        cmake_language += ['NONE']
    cmake_txt = textwrap.dedent('\n            cmake_minimum_required(VERSION ${{CMAKE_VERSION}})\n            project(MesonTemp LANGUAGES {})\n        ').format(' '.join(cmake_language)) + cmake_txt
    cm_file = build_dir / 'CMakeLists.txt'
    cm_file.write_text(cmake_txt, encoding='utf-8')
    mlog.cmd_ci_include(cm_file.absolute().as_posix())
    return build_dir