from __future__ import annotations
import abc
import os
import typing as T
import re
from .base import ArLikeLinker, RSPFileSyntax
from .. import mesonlib
from ..mesonlib import EnvironmentException, MesonException
from ..arglist import CompilerArgs
class CcrxDynamicLinker(DynamicLinker):
    """Linker for Renesas CCrx compiler."""
    id = 'rlink'

    def __init__(self, for_machine: mesonlib.MachineChoice, *, version: str='unknown version'):
        super().__init__(['rlink.exe'], for_machine, '', [], version=version)

    def get_accepts_rsp(self) -> bool:
        return False

    def get_lib_prefix(self) -> str:
        return '-lib='

    def get_std_shared_lib_args(self) -> T.List[str]:
        return []

    def get_output_args(self, outputname: str) -> T.List[str]:
        return [f'-output={outputname}']

    def get_search_args(self, dirname: str) -> 'T.NoReturn':
        raise OSError('rlink.exe does not have a search dir argument')

    def get_allow_undefined_args(self) -> T.List[str]:
        return []

    def get_soname_args(self, env: 'Environment', prefix: str, shlib_name: str, suffix: str, soversion: str, darwin_versions: T.Tuple[str, str]) -> T.List[str]:
        return []