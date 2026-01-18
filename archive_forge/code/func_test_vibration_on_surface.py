import os
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from ase import units, Atoms
import ase.io
from ase.calculators.qmmm import ForceConstantCalculator
from ase.vibrations import Vibrations, VibrationsData
from ase.thermochemistry import IdealGasThermo
def test_vibration_on_surface(self, testdir):
    from ase.build import fcc111, add_adsorbate
    ag_slab = fcc111('Ag', (4, 4, 2), a=2)
    n2 = Atoms('N2', positions=[[0.0, 0.0, 0.0], [0.0, np.sqrt(2), np.sqrt(2)]])
    add_adsorbate(ag_slab, n2, height=1, position='fcc')
    hessian_bottom_corner = np.zeros((2, 3, 2, 3))
    hessian_bottom_corner[-1, :, -2] = [1, 1, 1]
    hessian_bottom_corner[-2, :, -1] = [1, 1, 1]
    hessian = np.zeros((34, 3, 34, 3))
    hessian[32:, :, 32:, :] = hessian_bottom_corner
    ag_slab.calc = ForceConstantCalculator(hessian.reshape((34 * 3, 34 * 3)), ref=ag_slab.copy(), f0=np.zeros((34, 3)))
    vibs = Vibrations(ag_slab, indices=[-2, -1])
    vibs.run()
    vibs.read()
    assert_array_almost_equal(vibs.get_vibrations().get_hessian(), hessian_bottom_corner)
    vibs.summary()
    vibs.write_jmol()
    for i in range(6):
        assert_array_almost_equal(vibs.get_mode(i)[0], [0.0, 0.0, 0.0])
        assert np.all(vibs.get_mode(i)[-2:, :])