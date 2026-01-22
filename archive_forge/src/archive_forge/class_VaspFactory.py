import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
@factory('vasp')
class VaspFactory:

    def __init__(self, executable):
        self.executable = executable

    def version(self):
        from ase.calculators.vasp import get_vasp_version
        header = read_stdout([self.executable], createfile='INCAR')
        return get_vasp_version(header)

    def calc(self, **kwargs):
        from ase.calculators.vasp import Vasp
        if Vasp.VASP_PP_PATH not in os.environ:
            pytest.skip('No VASP pseudopotential path set. Set the ${} environment variable to enable.'.format(Vasp.VASP_PP_PATH))
        return Vasp(command=self.executable, **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['vasp'])