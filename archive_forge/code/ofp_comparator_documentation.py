import numpy as np
from itertools import combinations_with_replacement
from math import erf
from scipy.spatial.distance import cdist
from ase.neighborlist import NeighborList
from ase.utils import pbc2pbc
 Function for plotting all the individual fingerprints.
        Prefix = a prefix for the resulting PNG file.