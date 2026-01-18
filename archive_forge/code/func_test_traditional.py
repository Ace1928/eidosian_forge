import numpy as np
import pytest
from ase.vibrations import Vibrations
from ase.calculators.h2morse import (H2Morse, H2MorseCalculator,
from ase.calculators.h2morse import (H2MorseExcitedStatesCalculator,
def test_traditional(testdir):
    """Check that traditional calling works"""
    atoms = H2Morse()
    fname = 'exlist.dat'
    exl1 = H2MorseExcitedStatesAndCalculator(atoms.calc)
    exl1.write(fname)
    ex1 = exl1[0]
    exl2 = H2MorseExcitedStatesAndCalculator(fname, nstates=1)
    ex2 = exl2[-1]
    assert ex1.energy == pytest.approx(ex2.energy, 0.001)
    assert ex1.mur == pytest.approx(ex2.mur, 1e-05)
    assert ex1.muv == pytest.approx(ex2.muv, 1e-05)