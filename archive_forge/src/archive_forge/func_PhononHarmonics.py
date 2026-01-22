import numpy as np
from ase.parallel import world
from ase import units
from ase.md.md import process_temperature
def PhononHarmonics(atoms, force_constants, temp=None, *, temperature_K=None, rng=np.random, quantum=False, plus_minus=False, return_eigensolution=False, failfast=True):
    """Excite phonon modes to specified temperature.

    This will displace atomic positions and set the velocities so as
    to produce a random, phononically correct state with the requested
    temperature.

    Parameters:

    atoms: ase.atoms.Atoms() object
        Positions and momenta of this object are perturbed.

    force_constants: ndarray of size 3N x 3N
        Force constants for the the structure represented by atoms in eV/Å²

    temp: float (deprecated).
        Temperature in eV.  Deprecated, use ``temperature_K`` instead.

    temperature_K: float
        Temperature in Kelvin.

    rng: Random number generator
        RandomState or other random number generator, e.g., np.random.rand

    quantum: bool
        True for Bose-Einstein distribution, False for Maxwell-Boltzmann
        (classical limit)

    failfast: bool
        True for sanity checking the phonon spectrum for negative frequencies
        at Gamma.

    """
    d_ac, v_ac = phonon_harmonics(force_constants=force_constants, masses=atoms.get_masses(), temp=temp, temperature_K=temperature_K, rng=rng.rand, plus_minus=plus_minus, quantum=quantum, failfast=failfast, return_eigensolution=False)
    atoms.positions += d_ac
    atoms.set_velocities(v_ac)