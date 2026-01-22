from __future__ import annotations
from .common import CMakeException
from .generator import parse_generator_expressions
from .. import mlog
from ..mesonlib import version_compare
import typing as T
from pathlib import Path
from functools import lru_cache
import re
import json
import textwrap
def _cmake_add_custom_command(self, tline: CMakeTraceLine, name: T.Optional[str]=None) -> None:
    args = self._flatten_args(list(tline.args))
    if not args:
        return self._gen_exception('add_custom_command', 'requires at least 1 argument', tline)
    if args[0] == 'TARGET':
        return self._gen_exception('add_custom_command', 'TARGET syntax is currently not supported', tline)
    magic_keys = ['OUTPUT', 'COMMAND', 'MAIN_DEPENDENCY', 'DEPENDS', 'BYPRODUCTS', 'IMPLICIT_DEPENDS', 'WORKING_DIRECTORY', 'COMMENT', 'DEPFILE', 'JOB_POOL', 'VERBATIM', 'APPEND', 'USES_TERMINAL', 'COMMAND_EXPAND_LISTS']
    target = CMakeGeneratorTarget(name)

    def handle_output(key: str, target: CMakeGeneratorTarget) -> None:
        target._outputs_str += [key]

    def handle_command(key: str, target: CMakeGeneratorTarget) -> None:
        if key == 'ARGS':
            return
        target.command[-1] += [key]

    def handle_depends(key: str, target: CMakeGeneratorTarget) -> None:
        target.depends += [key]
    working_dir = None

    def handle_working_dir(key: str, target: CMakeGeneratorTarget) -> None:
        nonlocal working_dir
        if working_dir is None:
            working_dir = key
        else:
            working_dir += ' '
            working_dir += key
    fn = None
    for i in args:
        if i in magic_keys:
            if i == 'OUTPUT':
                fn = handle_output
            elif i == 'DEPENDS':
                fn = handle_depends
            elif i == 'WORKING_DIRECTORY':
                fn = handle_working_dir
            elif i == 'COMMAND':
                fn = handle_command
                target.command += [[]]
            else:
                fn = None
            continue
        if fn is not None:
            fn(i, target)
    cbinary_dir = self.var_to_str('MESON_PS_CMAKE_CURRENT_BINARY_DIR')
    csource_dir = self.var_to_str('MESON_PS_CMAKE_CURRENT_SOURCE_DIR')
    target.working_dir = Path(working_dir) if working_dir else None
    target.current_bin_dir = Path(cbinary_dir) if cbinary_dir else None
    target.current_src_dir = Path(csource_dir) if csource_dir else None
    target._outputs_str = self._guess_files(target._outputs_str)
    target.depends = self._guess_files(target.depends)
    target.command = [self._guess_files(x) for x in target.command]
    self.custom_targets += [target]
    if name:
        self.targets[name] = target