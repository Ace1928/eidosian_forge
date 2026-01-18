import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
Make the arrays used to store data about the atoms.

        In a parallel simulation, these are migrating arrays.  In a
        serial simulation they are ordinary Numeric arrays.
        