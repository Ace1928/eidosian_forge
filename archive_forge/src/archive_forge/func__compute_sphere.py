import collections
import math
import numpy as np
from Bio.PDB.kdtrees import KDTree
def _compute_sphere(self):
    """Return the 3D coordinates of n points on a sphere.

        Uses the golden spiral algorithm to place points 'evenly' on the sphere
        surface. We compute this once and then move the sphere to the centroid
        of each atom as we compute the ASAs.
        """
    n = self.n_points
    dl = np.pi * (3 - 5 ** 0.5)
    dz = 2.0 / n
    longitude = 0
    z = 1 - dz / 2
    coords = np.zeros((n, 3), dtype=np.float32)
    for k in range(n):
        r = (1 - z * z) ** 0.5
        coords[k, 0] = math.cos(longitude) * r
        coords[k, 1] = math.sin(longitude) * r
        coords[k, 2] = z
        z -= dz
        longitude += dl
    return coords