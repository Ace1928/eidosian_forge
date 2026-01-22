import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
@factory('cp2k')
class CP2KFactory:

    def __init__(self, executable):
        self.executable = executable

    def version(self):
        from ase.calculators.cp2k import Cp2kShell
        shell = Cp2kShell(self.executable, debug=False)
        return shell.version

    def calc(self, **kwargs):
        from ase.calculators.cp2k import CP2K
        return CP2K(command=self.executable, **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return CP2KFactory(config.executables['cp2k'])