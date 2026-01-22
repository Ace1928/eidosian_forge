from math import sqrt
from operator import itemgetter
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.data import reference_states, atomic_numbers
from ase.lattice.cubic import FaceCenteredCubic
def add_vacuum(atoms, vacuum):
    """Add vacuum layer to the atoms.

    Parameters:

    atoms: Atoms object
        Most likely created by one of the surface functions.
    vacuum: float
        The thickness of the vacuum layer (in Angstrom).
    """
    uc = atoms.get_cell()
    normal = np.cross(uc[0], uc[1])
    costheta = np.dot(normal, uc[2]) / np.sqrt(np.dot(normal, normal) * np.dot(uc[2], uc[2]))
    length = np.sqrt(np.dot(uc[2], uc[2]))
    newlength = length + vacuum / costheta
    uc[2] *= newlength / length
    atoms.set_cell(uc)