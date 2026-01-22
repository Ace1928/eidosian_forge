from __future__ import annotations
import abc
import os
import typing as T
import re
from .base import ArLikeLinker, RSPFileSyntax
from .. import mesonlib
from ..mesonlib import EnvironmentException, MesonException
from ..arglist import CompilerArgs
class MSVCDynamicLinker(VisualStudioLikeLinkerMixin, DynamicLinker):
    """Microsoft's Link.exe."""
    id = 'link'

    def __init__(self, for_machine: mesonlib.MachineChoice, always_args: T.List[str], *, exelist: T.Optional[T.List[str]]=None, prefix: T.Union[str, T.List[str]]='', machine: str='x86', version: str='unknown version', direct: bool=True):
        super().__init__(exelist or ['link.exe'], for_machine, prefix, always_args, machine=machine, version=version, direct=direct)

    def get_always_args(self) -> T.List[str]:
        return self._apply_prefix(['/release']) + super().get_always_args()

    def get_win_subsystem_args(self, value: str) -> T.List[str]:
        return self._apply_prefix([f'/SUBSYSTEM:{value.upper()}'])