import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.optimize.precon import Exp, PreconLBFGS
@pytest.mark.slow
def test_precon_amin():
    cu0 = bulk('Cu') * (2, 2, 2)
    sigma = cu0.get_distance(0, 1) * 2.0 ** (-1.0 / 6)
    lj = LennardJones(sigma=sigma)
    cell = cu0.get_cell()
    cell *= 0.95
    cell[1, 0] += 0.2
    cell[2, 1] += 0.5
    cu0.set_cell(cell, scale_atoms=True)
    energies = []
    for use_armijo in [True, False]:
        for a_min in [None, 0.001]:
            atoms = cu0.copy()
            atoms.calc = lj
            opt = PreconLBFGS(atoms, precon=Exp(A=3), use_armijo=use_armijo, a_min=a_min, variable_cell=True)
            opt.run(fmax=0.001, smax=0.0001)
            energies.append(atoms.get_potential_energy())
    assert np.abs(np.array(energies) - -63.5032311942).max() < 0.0001