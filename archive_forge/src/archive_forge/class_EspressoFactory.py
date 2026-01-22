import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
@factory('espresso')
class EspressoFactory:

    def __init__(self, executable, pseudo_dir):
        self.executable = executable
        self.pseudo_dir = pseudo_dir

    def _base_kw(self):
        from ase.units import Ry
        return dict(ecutwfc=300 / Ry)

    def version(self):
        stdout = read_stdout([self.executable])
        match = re.match('\\s*Program PWSCF\\s*(\\S+)', stdout, re.M)
        assert match is not None
        return match.group(1)

    def calc(self, **kwargs):
        from ase.calculators.espresso import Espresso
        command = '{} -in PREFIX.pwi > PREFIX.pwo'.format(self.executable)
        pseudopotentials = {}
        for path in self.pseudo_dir.glob('*.UPF'):
            fname = path.name
            symbol = fname.split('_', 1)[0].capitalize()
            pseudopotentials[symbol] = fname
        kw = self._base_kw()
        kw.update(kwargs)
        return Espresso(command=command, pseudo_dir=str(self.pseudo_dir), pseudopotentials=pseudopotentials, **kw)

    @classmethod
    def fromconfig(cls, config):
        paths = config.datafiles['espresso']
        assert len(paths) == 1
        return cls(config.executables['espresso'], paths[0])