import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
@factory('elk')
class ElkFactory:

    def __init__(self, executable, species_dir):
        self.executable = executable
        self.species_dir = species_dir

    def version(self):
        output = read_stdout([self.executable])
        match = re.search('Elk code version (\\S+)', output, re.M)
        return match.group(1)

    def calc(self, **kwargs):
        from ase.calculators.elk import ELK
        command = f'{self.executable} > elk.out'
        return ELK(command=command, species_dir=self.species_dir, **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['elk'], config.datafiles['elk'][0])