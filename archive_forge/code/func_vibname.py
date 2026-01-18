import sys
import numpy as np
from math import factorial
from pytest import approx, fixture
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from ase.vibrations.franck_condon import (FranckCondonOverlap,
@fixture()
def vibname(testdir, relaxed):
    atoms = relaxed.copy()
    atoms.calc = relaxed.calc
    name = 'vib'
    vib = Vibrations(atoms, name=name)
    vib.run()
    return name