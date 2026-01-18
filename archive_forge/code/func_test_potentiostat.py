import pytest
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.contour_exploration import ContourExploration
import numpy as np
from ase.calculators.emt import EMT
from .test_ce_curvature import Al_atom_pair
def test_potentiostat(testdir):
    """This is very realistic and stringent test of the potentiostatic accuracy
     with 32 atoms at ~235 meV/atom above the ground state."""
    name = 'test_potentiostat'
    seed = 19460926
    atoms = Al_block()
    E0 = atoms.get_potential_energy()
    atoms.rattle(stdev=0.18, seed=seed)
    initial_energy = atoms.get_potential_energy()
    rng = np.random.RandomState(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=rng)
    with ContourExploration(atoms, **bulk_Al_settings, energy_target=initial_energy, rng=rng, trajectory=name + '.traj', logfile=name + '.log') as dyn:
        print('Energy Above Ground State: {: .4f} eV/atom'.format((initial_energy - E0) / len(atoms)))
        for i in range(5):
            dyn.run(5)
            energy_error = (atoms.get_potential_energy() - initial_energy) / len(atoms)
            print('Potentiostat Error {: .4f} eV/atom'.format(energy_error))
            assert 0 == pytest.approx(energy_error, abs=0.01)