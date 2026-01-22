from __future__ import annotations
import abc
import os
import typing as T
import re
from .base import ArLikeLinker, RSPFileSyntax
from .. import mesonlib
from ..mesonlib import EnvironmentException, MesonException
from ..arglist import CompilerArgs
class AIXDynamicLinker(PosixDynamicLinkerMixin, DynamicLinker):
    """Sys-V derived linker used on AIX"""
    id = 'ld.aix'

    def get_always_args(self) -> T.List[str]:
        return self._apply_prefix(['-bnoipath', '-bbigtoc']) + super().get_always_args()

    def no_undefined_args(self) -> T.List[str]:
        return self._apply_prefix(['-bernotok'])

    def get_allow_undefined_args(self) -> T.List[str]:
        return self._apply_prefix(['-berok'])

    def get_archive_name(self, filename: str) -> str:
        filename = re.sub('[.][a]([.]?([0-9]+))*([.]?([a-z]+))*', '.a', filename.replace('.so', '.a'))
        return filename

    def get_command_to_archive_shlib(self) -> T.List[str]:
        command = ['ar', '-q', '-v', '$out', '$in', '&&', 'rm', '-f', '$in']
        return command

    def get_link_whole_for(self, args: T.List[str]) -> T.List[str]:
        return args

    def build_rpath_args(self, env: 'Environment', build_dir: str, from_dir: str, rpath_paths: T.Tuple[str, ...], build_rpath: str, install_rpath: str) -> T.Tuple[T.List[str], T.Set[bytes]]:
        all_paths: mesonlib.OrderedSet[str] = mesonlib.OrderedSet()
        if install_rpath != '':
            all_paths.add(install_rpath)
        if build_rpath != '':
            all_paths.add(build_rpath)
        for p in rpath_paths:
            all_paths.add(os.path.join(build_dir, p))
        sys_path = env.get_compiler_system_lib_dirs(self.for_machine)
        if len(sys_path) == 0:
            all_paths.update(['/usr/lib', '/lib'])
        else:
            for p in sys_path:
                if os.path.isdir(p):
                    all_paths.add(p)
        return (self._apply_prefix('-blibpath:' + ':'.join(all_paths)), set())

    def thread_flags(self, env: 'Environment') -> T.List[str]:
        return ['-pthread']