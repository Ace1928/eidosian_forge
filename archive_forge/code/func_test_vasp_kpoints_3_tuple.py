import os
import pytest
from ase.build import bulk
from .filecmp_ignore_whitespace import filecmp_ignore_whitespace
@calc('vasp')
def test_vasp_kpoints_3_tuple(factory, write_kpoints):
    write_kpoints(factory, gamma=False, kpts=(4, 4, 4))
    check_kpoints_line(2, 'Monkhorst-Pack')
    check_kpoints_line(3, '4 4 4')