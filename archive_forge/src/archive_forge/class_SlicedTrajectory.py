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
class SlicedTrajectory:
    """Wrapper to return a slice from a trajectory without loading
    from disk. Initialize with a trajectory (in read mode) and the
    desired slice object."""

    def __init__(self, trajectory, sliced):
        self.trajectory = trajectory
        self.map = range(len(self.trajectory))[sliced]

    def __getitem__(self, i):
        if isinstance(i, slice):
            traj = SlicedTrajectory(self.trajectory, slice(0, None))
            traj.map = self.map[i]
            return traj
        return self.trajectory[self.map[i]]

    def __len__(self):
        return len(self.map)