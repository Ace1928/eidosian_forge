import os
import pytest
from ase.build import bulk
from .filecmp_ignore_whitespace import filecmp_ignore_whitespace
@calc('vasp')
def test_vasp_kpoints_111(factory, write_kpoints):
    write_kpoints(factory, gamma=True)
    check_kpoints_line(2, 'Gamma')
    check_kpoints_line(3, '1 1 1')