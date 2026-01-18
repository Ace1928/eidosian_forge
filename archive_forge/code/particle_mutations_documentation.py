import numpy as np
from operator import itemgetter
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import get_distance_matrix, get_nndist
from ase import Atoms
Does the actual substitution