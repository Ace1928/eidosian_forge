import os
import pytest
from ase.build import bulk
from .filecmp_ignore_whitespace import filecmp_ignore_whitespace
def _write_kpoints(factory, **kwargs):
    calc = factory.calc(**kwargs)
    calc.initialize(atoms)
    calc.write_kpoints(atoms=atoms)
    return (atoms, calc)