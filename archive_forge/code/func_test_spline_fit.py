import json
import numpy as np
import pytest
from ase.calculators.morse import MorsePotential
from ase.optimize import BFGS, ODE12r
from ase.optimize.precon import Exp
from ase.build import bulk
from ase.neb import NEB, NEBTools, NEBOptimizer
from ase.geometry.geometry import find_mic
from ase.constraints import FixBondLength
from ase.geometry.geometry import get_distances
from ase.utils.forcecurve import fit_images
def test_spline_fit(setup_images):
    images, _, _ = setup_images
    neb = NEB(images)
    fit = neb.spline_fit()
    assert np.allclose(fit.s, np.linspace(0, 1, len(images)))
    assert np.allclose(fit.x(fit.s), fit.x_data)
    eps = 0.0001
    assert np.allclose(fit.dx_ds(fit.s[2] + eps), fit.dx_ds(fit.s[2] + eps))