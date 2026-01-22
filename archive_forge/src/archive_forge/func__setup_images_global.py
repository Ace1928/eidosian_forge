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
@pytest.fixture(scope='module')
def _setup_images_global():
    N_intermediate = 3
    N_cell = 2
    initial = bulk('Cu', cubic=True)
    initial *= N_cell
    D, D_len = get_distances(np.diag(initial.cell) / 2, initial.positions, initial.cell, initial.pbc)
    vac_index = D_len.argmin()
    vac_pos = initial.positions[vac_index]
    del initial[vac_index]
    D, D_len = get_distances(vac_pos, initial.positions, initial.cell, initial.pbc)
    D = D[0, :]
    D_len = D_len[0, :]
    nn_mask = np.abs(D_len - D_len.min()) < 1e-08
    i1 = nn_mask.nonzero()[0][0]
    i2 = ((D + D[i1]) ** 2).sum(axis=1).argmin()
    print(f'vac_index={vac_index} i1={i1} i2={i2} distance={initial.get_distance(i1, i2, mic=True)}')
    final = initial.copy()
    final.positions[i1] = vac_pos
    initial.calc = calc()
    final.calc = calc()
    qn = ODE12r(initial)
    qn.run(fmax=0.001)
    qn = ODE12r(final)
    qn.run(fmax=0.001)
    images = [initial]
    for image in range(N_intermediate):
        image = initial.copy()
        image.calc = calc()
        images.append(image)
    images.append(final)
    neb = NEB(images)
    neb.interpolate()
    return (neb.images, i1, i2)