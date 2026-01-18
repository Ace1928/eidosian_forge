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
def write_traj(fd, images):
    """Write image(s) to trajectory."""
    trj = TrajectoryWriter(fd)
    if isinstance(images, Atoms):
        images = [images]
    for atoms in images:
        trj.write(atoms)