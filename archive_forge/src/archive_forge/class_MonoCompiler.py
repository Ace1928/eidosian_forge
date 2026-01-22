from __future__ import annotations
import os.path, subprocess
import textwrap
import typing as T
from ..mesonlib import EnvironmentException
from ..linkers import RSPFileSyntax
from .compilers import Compiler
from .mixins.islinker import BasicLinkerIsCompilerMixin
class MonoCompiler(CsCompiler):
    id = 'mono'

    def __init__(self, exelist: T.List[str], version: str, for_machine: MachineChoice, info: 'MachineInfo'):
        super().__init__(exelist, version, for_machine, info, runner='mono')

    def rsp_file_syntax(self) -> 'RSPFileSyntax':
        return RSPFileSyntax.GCC