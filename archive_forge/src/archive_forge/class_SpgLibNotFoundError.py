from collections import Counter
from itertools import combinations, product, filterfalse
import numpy as np
from scipy.spatial import cKDTree as KDTree
from ase import Atom, Atoms
from ase.build.tools import niggli_reduce
class SpgLibNotFoundError(Exception):
    """Raised if SPG lib is not found when needed."""

    def __init__(self, msg):
        super(SpgLibNotFoundError, self).__init__(msg)