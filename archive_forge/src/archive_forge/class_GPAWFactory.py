import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
@factory('gpaw')
class GPAWFactory:
    importname = 'gpaw'

    def calc(self, **kwargs):
        from gpaw import GPAW
        return GPAW(**kwargs)

    def version(self):
        import gpaw
        return gpaw.__version__

    @classmethod
    def fromconfig(cls, config):
        import importlib
        spec = importlib.util.find_spec('gpaw')
        if spec is None:
            raise NotInstalled('gpaw')
        return cls()