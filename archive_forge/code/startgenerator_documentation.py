import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.build import molecule
from ase.ga.utilities import (closest_distances_generator, atoms_too_close,
Generates a random unit cell.

        For this, we use the vectors in self.slab.cell
        in the fixed directions and randomly generate
        the variable ones. For such a cell to be valid,
        it has to satisfy the self.cellbounds constraints.

        The cell will also be such that the volume of the
        box in which the atoms can be placed (box limits
        described by self.box_to_place_in) is equal to
        self.box_volume.

        Parameters:

        repeat: tuple of 3 integers
            Indicates by how much each cell vector
            will later be reduced by cell splitting.

            This is used to ensure that the original
            cell is large enough so that the cell lengths
            of the smaller cell exceed the largest
            (X,X)-minimal-interatomic-distance in self.blmin.
        