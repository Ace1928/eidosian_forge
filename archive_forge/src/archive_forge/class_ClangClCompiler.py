from __future__ import annotations
import abc
import os
import typing as T
from ... import arglist
from ... import mesonlib
from ... import mlog
from mesonbuild.compilers.compilers import CompileCheckMode
class ClangClCompiler(VisualStudioLikeCompiler):
    """Specific to Clang-CL."""
    id = 'clang-cl'

    def __init__(self, target: str):
        super().__init__(target)
        self.can_compile_suffixes.add('s')
        self.can_compile_suffixes.add('sx')

    def has_arguments(self, args: T.List[str], env: 'Environment', code: str, mode: CompileCheckMode) -> T.Tuple[bool, bool]:
        if mode != CompileCheckMode.LINK:
            args = args + ['-Werror=unknown-argument', '-Werror=unknown-warning-option']
        return super().has_arguments(args, env, code, mode)

    def get_toolset_version(self) -> T.Optional[str]:
        return '14.1'

    def get_pch_base_name(self, header: str) -> str:
        return header

    def get_include_args(self, path: str, is_system: bool) -> T.List[str]:
        if path == '':
            path = '.'
        return ['/clang:-isystem' + path] if is_system else ['-I' + path]

    def get_dependency_compile_args(self, dep: 'Dependency') -> T.List[str]:
        if dep.get_include_type() == 'system':
            converted: T.List[str] = []
            for i in dep.get_compile_args():
                if i.startswith('-isystem'):
                    converted += ['/clang:' + i]
                else:
                    converted += [i]
            return converted
        else:
            return dep.get_compile_args()