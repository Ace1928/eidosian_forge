import numpy as np
from math import cos, sin, pi
from ase.calculators.lammpslib import convert_cell
from ase.ga.utilities import (atoms_too_close,
from ase.ga.offspring_creator import OffspringCreator, CombinationMutation
from ase import Atoms
class RattleRotationalMutation(CombinationMutation):
    """Combination of RattleMutation and RotationalMutation.

    Parameters:

    rattlemutation: OffspringCreator instance
        A mutation that rattles atoms.

    rotationalmutation: OffspringCreator instance
        A mutation that rotates moieties.
    """

    def __init__(self, rattlemutation, rotationalmutation, verbose=False):
        super(RattleRotationalMutation, self).__init__(rattlemutation, rotationalmutation, verbose=verbose)
        self.descriptor = 'rattlerotational'