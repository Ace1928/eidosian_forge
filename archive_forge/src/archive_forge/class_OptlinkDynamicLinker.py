from __future__ import annotations
import abc
import os
import typing as T
import re
from .base import ArLikeLinker, RSPFileSyntax
from .. import mesonlib
from ..mesonlib import EnvironmentException, MesonException
from ..arglist import CompilerArgs
class OptlinkDynamicLinker(VisualStudioLikeLinkerMixin, DynamicLinker):
    """Digital Mars dynamic linker for windows."""
    id = 'optlink'

    def __init__(self, exelist: T.List[str], for_machine: mesonlib.MachineChoice, *, version: str='unknown version'):
        super().__init__(exelist, for_machine, '', [], version=version)

    def get_allow_undefined_args(self) -> T.List[str]:
        return []

    def get_debugfile_args(self, targetfile: str) -> T.List[str]:
        return []

    def get_always_args(self) -> T.List[str]:
        return []