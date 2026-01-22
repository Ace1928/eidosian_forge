import itertools
import numpy as np
from ase.geometry.dimensionality.disjoint_set import DisjointSet
def _get_component_dimensionalities(self):
    n = self.n
    offsets = self.offsets
    single_roots = np.unique(self.gsingle.find_all())
    super_components = self.gsuper.find_all()
    component_dim = {}
    for i in single_roots:
        num_clusters = len(np.unique(super_components[offsets + i]))
        dim = {n ** 3: 0, n ** 2: 1, n: 2, 1: 3}[num_clusters]
        component_dim[i] = dim
    return component_dim