import warnings
from typing import Tuple
import numpy as np
from ase import __version__
from ase.calculators.singlepoint import SinglePointCalculator, all_properties
from ase.constraints import dict2constraint
from ase.calculators.calculator import PropertyNotImplementedError
from ase.atoms import Atoms
from ase.io.jsonio import encode, decode
from ase.io.pickletrajectory import PickleTrajectory
from ase.parallel import world
from ase.utils import tokenize_version
def write_atoms(backend, atoms, write_header=True):
    b = backend
    if write_header:
        b.write(pbc=atoms.pbc.tolist(), numbers=atoms.numbers)
        if atoms.constraints:
            if all((hasattr(c, 'todict') for c in atoms.constraints)):
                b.write(constraints=encode(atoms.constraints))
        if atoms.has('masses'):
            b.write(masses=atoms.get_masses())
    b.write(positions=atoms.get_positions(), cell=atoms.get_cell().tolist())
    if atoms.has('tags'):
        b.write(tags=atoms.get_tags())
    if atoms.has('momenta'):
        b.write(momenta=atoms.get_momenta())
    if atoms.has('initial_magmoms'):
        b.write(magmoms=atoms.get_initial_magnetic_moments())
    if atoms.has('initial_charges'):
        b.write(charges=atoms.get_initial_charges())