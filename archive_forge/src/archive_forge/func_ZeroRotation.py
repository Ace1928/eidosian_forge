import numpy as np
from ase.parallel import world
from ase import units
from ase.md.md import process_temperature
def ZeroRotation(atoms, preserve_temperature=True):
    """Sets the total angular momentum to zero by counteracting rigid rotations."""
    temp0 = atoms.get_temperature()
    Ip, basis = atoms.get_moments_of_inertia(vectors=True)
    Lp = np.dot(basis, atoms.get_angular_momentum())
    omega = np.dot(np.linalg.inv(basis), np.select([Ip > 0], [Lp / Ip]))
    com = atoms.get_center_of_mass()
    positions = atoms.get_positions()
    positions -= com
    velocities = atoms.get_velocities()
    atoms.set_velocities(velocities - np.cross(omega, positions))
    if preserve_temperature:
        force_temperature(atoms, temp0)