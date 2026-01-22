import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
@factory('siesta')
class SiestaFactory:

    def __init__(self, executable, pseudo_path):
        self.executable = executable
        self.pseudo_path = pseudo_path

    def version(self):
        from ase.calculators.siesta.siesta import get_siesta_version
        full_ver = get_siesta_version(self.executable)
        m = re.match('siesta-(\\S+)', full_ver, flags=re.I)
        if m:
            return m.group(1)
        return full_ver

    def calc(self, **kwargs):
        from ase.calculators.siesta import Siesta
        command = '{} < PREFIX.fdf > PREFIX.out'.format(self.executable)
        return Siesta(command=command, pseudo_path=str(self.pseudo_path), **kwargs)

    @classmethod
    def fromconfig(cls, config):
        paths = config.datafiles['siesta']
        assert len(paths) == 1
        path = paths[0]
        return cls(config.executables['siesta'], str(path))