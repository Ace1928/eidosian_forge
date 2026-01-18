import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.constraints import UnitCellFilter
from ase.optimize import BFGS
def test_stress_voigt_shape(atoms):
    for ideal_gas in (False, True):
        kw = {'include_ideal_gas': ideal_gas}
        assert atoms.get_stress(voigt=True, **kw).shape == (6,)
        assert atoms.get_stress(voigt=False, **kw).shape == (3, 3)
        assert atoms.get_stresses(voigt=True, **kw).shape == (len(atoms), 6)
        assert atoms.get_stresses(voigt=False, **kw).shape == (len(atoms), 3, 3)