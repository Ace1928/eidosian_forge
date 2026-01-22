from typing import List, Optional
import numpy as np
from ase.data import atomic_numbers as ref_atomic_numbers
from ase.spacegroup import Spacegroup
from ase.cluster.base import ClusterBase
from ase.cluster.cluster import Cluster
def GCD(a, b):
    """Greatest Common Divisor of a and b."""
    while a != 0:
        a, b = (b % a, a)
    return b