from ase.ga.offspring_creator import OffspringCreator
from ase import Atoms
from itertools import chain
import numpy as np
Generator function that returns each vector (between atoms)
        that is shorter than the minimum distance for those atom types
        (set during the initialization in blmin).