import pytest
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.contour_exploration import ContourExploration
import numpy as np
from ase.calculators.emt import EMT
from .test_ce_curvature import Al_atom_pair
def test_potentiostat_no_fs(testdir):
    """This test ensures that the potentiostat is working even when curvature
    extrapolation (use_fs) is turned off."""
    name = 'test_potentiostat_no_fs'
    atoms = Al_atom_pair()
    atoms.set_momenta([[0, -1, 0], [0, 1, 0]])
    initial_energy = atoms.get_potential_energy()
    with ContourExploration(atoms, maxstep=0.2, parallel_drift=0.0, remove_translation=False, energy_target=initial_energy, potentiostat_step_scale=None, use_frenet_serret=False, trajectory=name + '.traj', logfile=name + '.log') as dyn:
        for i in range(5):
            dyn.run(10)
            energy_error = (atoms.get_potential_energy() - initial_energy) / len(atoms)
            print('Potentiostat Error {: .4f} eV/atom'.format(energy_error))
            assert 0 == pytest.approx(energy_error, abs=0.01)