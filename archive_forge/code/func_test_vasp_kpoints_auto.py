import os
import pytest
from ase.build import bulk
from .filecmp_ignore_whitespace import filecmp_ignore_whitespace
@calc('vasp')
def test_vasp_kpoints_auto(factory, write_kpoints):
    write_kpoints(factory, kpts=20)
    check_kpoints_line(1, '0')
    check_kpoints_line(2, 'Auto')
    check_kpoints_line(3, '20')