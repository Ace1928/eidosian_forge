import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
@factory('nwchem')
class NWChemFactory:

    def __init__(self, executable):
        self.executable = executable

    def version(self):
        stdout = read_stdout([self.executable], createfile='nwchem.nw')
        match = re.search('Northwest Computational Chemistry Package \\(NWChem\\) (\\S+)', stdout, re.M)
        return match.group(1)

    def calc(self, **kwargs):
        from ase.calculators.nwchem import NWChem
        command = f'{self.executable} PREFIX.nwi > PREFIX.nwo'
        return NWChem(command=command, **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['nwchem'])