from pathlib import Path
import numpy as np
import pytest
import ase.io
from ase.io import extxyz
from ase.atoms import Atoms
from ase.build import bulk
from ase.io.extxyz import escape
from ase.calculators.calculator import compare_atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms, FixCartesian
from ase.stress import full_3x3_to_voigt_6_stress
from ase.build import molecule
def test_vec_cell(at, images):
    ase.io.write('multi.xyz', images, vec_cell=True)
    cell = images[1].get_cell()
    cell[-1] = [0.0, 0.0, 0.0]
    images[1].set_cell(cell)
    cell = images[2].get_cell()
    cell[-1] = [0.0, 0.0, 0.0]
    cell[-2] = [0.0, 0.0, 0.0]
    images[2].set_cell(cell)
    read_images = ase.io.read('multi.xyz', index=':')
    assert read_images == images
    Path('structure.xyz').write_text('1\n    Coordinates\n    C         -7.28250        4.71303       -3.82016\n      VEC1 1.0 0.1 1.1\n    1\n\n    C         -7.28250        4.71303       -3.82016\n    VEC1 1.0 0.1 1.1\n    ')
    a = ase.io.read('structure.xyz', index=0)
    b = ase.io.read('structure.xyz', index=1)
    assert a == b
    Path('structure.xyz').write_text('4\n    Coordinates\n    MG        -4.25650        3.79180       -2.54123\n    C         -1.15405        2.86652       -1.26699\n    C         -5.53758        3.70936        0.63504\n    C         -7.28250        4.71303       -3.82016\n\n    ')
    a = ase.io.read('structure.xyz')
    assert a[0].symbol == 'Mg'