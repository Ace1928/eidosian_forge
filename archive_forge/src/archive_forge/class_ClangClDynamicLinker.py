from __future__ import annotations
import abc
import os
import typing as T
import re
from .base import ArLikeLinker, RSPFileSyntax
from .. import mesonlib
from ..mesonlib import EnvironmentException, MesonException
from ..arglist import CompilerArgs
class ClangClDynamicLinker(VisualStudioLikeLinkerMixin, DynamicLinker):
    """Clang's lld-link.exe."""
    id = 'lld-link'

    def __init__(self, for_machine: mesonlib.MachineChoice, always_args: T.List[str], *, exelist: T.Optional[T.List[str]]=None, prefix: T.Union[str, T.List[str]]='', machine: str='x86', version: str='unknown version', direct: bool=True):
        super().__init__(exelist or ['lld-link.exe'], for_machine, prefix, always_args, machine=machine, version=version, direct=direct)

    def get_output_args(self, outputname: str) -> T.List[str]:
        if self.machine is None:
            return self._apply_prefix([f'/OUT:{outputname}'])
        return super().get_output_args(outputname)

    def get_win_subsystem_args(self, value: str) -> T.List[str]:
        return self._apply_prefix([f'/SUBSYSTEM:{value.upper()}'])

    def get_thinlto_cache_args(self, path: str) -> T.List[str]:
        return ['/lldltocache:' + path]