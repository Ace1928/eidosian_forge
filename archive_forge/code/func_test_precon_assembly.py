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
def test_precon_assembly(setup_images):
    images, _, _ = setup_images
    neb = NEB(images, method='spline', precon='Exp')
    neb.get_forces()
    for image, precon in zip(neb.images, neb.precon):
        assert isinstance(precon, Exp)
        P = precon.asarray()
        N = 3 * len(image)
        assert P.shape == (N, N)
        assert np.abs(P - P.T).max() < 1e-06
        assert np.all(np.linalg.eigvalsh(P)) > 0