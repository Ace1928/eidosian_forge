from collections import defaultdict
import numpy as np
import kimpy
from kimpy import neighlist
from ase.neighborlist import neighbor_list
from ase import Atom
from .kimpy_wrappers import check_call_wrapper
Create the neighbor list along with the other required
        parameters (which are stored as instance attributes). The
        required parameters are:

            - num_particles
            - coords
            - particle_contributing
            - species_code

        Note that the KIM API requires a neighbor list that has indices
        corresponding to each atom.
        