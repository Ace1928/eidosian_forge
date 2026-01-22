import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.ga.utilities import (atoms_too_close, atoms_too_close_two_sets,
from ase.ga.offspring_creator import OffspringCreator
Creates a child from two parents using the given cut.

        Returns None if the generated structure does not contain
        a large enough fraction of each parent (see self.minfrac).

        Does not check whether atoms are too close.

        Assumes the 'slab' parts have been removed from the parent
        structures and that these have been checked for equal
        lengths, stoichiometries, and tags (if self.use_tags).

        Parameters:

        cutting_normal: int or (1x3) array

        cutting_point: (1x3) array
            In fractional coordinates

        cell: (3x3) array
            The unit cell for the child structure
        