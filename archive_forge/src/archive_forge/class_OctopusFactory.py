import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
@factory('octopus')
class OctopusFactory:

    def __init__(self, executable):
        self.executable = executable

    def version(self):
        stdout = read_stdout([self.executable, '--version'])
        match = re.match('octopus\\s*(.+)', stdout)
        return match.group(1)

    def calc(self, **kwargs):
        from ase.calculators.octopus import Octopus
        command = f'{self.executable} > stdout.log'
        return Octopus(command=command, **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['octopus'])