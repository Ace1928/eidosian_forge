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
def test_read_slash():
    Path('slash.xyz').write_text('4\n    key1=a key2=a/b key3=a@b key4="a@b"\n    Mg        -4.25650        3.79180       -2.54123\n    C         -1.15405        2.86652       -1.26699\n    C         -5.53758        3.70936        0.63504\n    C         -7.28250        4.71303       -3.82016\n    ')
    a = ase.io.read('slash.xyz')
    assert a.info['key1'] == 'a'
    assert a.info['key2'] == 'a/b'
    assert a.info['key3'] == 'a@b'
    assert a.info['key4'] == 'a@b'