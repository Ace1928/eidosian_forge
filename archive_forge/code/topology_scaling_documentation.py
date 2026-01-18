import itertools
import numpy as np
from ase.geometry.dimensionality.disjoint_set import DisjointSet
Determines the dimensionality and constituent atoms of each
        component.

        Returns:
        components: array    The component ID every atom
        