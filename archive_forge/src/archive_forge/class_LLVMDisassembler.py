import argparse
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
class LLVMDisassembler:
    _path: str
    _ll_file: str

    def __init__(self, path) -> None:
        """
        Invoke llvm-dis to disassemble the given file.
        :param path: path to llvm-dis
        """
        self._path = path
        self._ll_file = '/tmp/extern_lib.ll'

    def disasm(self, lib_path: str) -> None:
        subprocess.Popen([self._path, lib_path, '-o', self.ll_file], stdout=subprocess.PIPE).communicate()

    @property
    def ll_file(self) -> str:
        return self._ll_file

    @property
    def path(self) -> str:
        return self._path