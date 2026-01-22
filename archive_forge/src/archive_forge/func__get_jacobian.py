import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation
from matplotlib.tri._trifinder import TriFinder
from matplotlib.tri._tritools import TriAnalyzer
@staticmethod
def _get_jacobian(tris_pts):
    """
        Fast (vectorized) function to compute triangle jacobian matrix.

        Parameters
        ----------
        tris_pts : array like of dim 3 (shape: (nx, 3, 2))
            Coordinates of the containing triangles apexes.

        Returns
        -------
        array of dim 3 (shape (nx, 2, 2))
            Barycentric coordinates of the points inside the containing
            triangles.
            J[itri, :, :] is the jacobian matrix at apex 0 of the triangle
            itri, so that the following (matrix) relationship holds:
               [dz/dksi] = [J] x [dz/dx]
            with x: global coordinates
                 ksi: element parametric coordinates in triangle first apex
                 local basis.
        """
    a = np.array(tris_pts[:, 1, :] - tris_pts[:, 0, :])
    b = np.array(tris_pts[:, 2, :] - tris_pts[:, 0, :])
    J = _to_matrix_vectorized([[a[:, 0], a[:, 1]], [b[:, 0], b[:, 1]]])
    return J