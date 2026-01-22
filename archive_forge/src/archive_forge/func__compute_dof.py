import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation
from matplotlib.tri._trifinder import TriFinder
from matplotlib.tri._tritools import TriAnalyzer
def _compute_dof(self, kind, dz=None):
    """
        Compute and return nodal dofs according to kind.

        Parameters
        ----------
        kind : {'min_E', 'geom', 'user'}
            Choice of the _DOF_estimator subclass to estimate the gradient.
        dz : tuple of array-likes (dzdx, dzdy), optional
            Used only if *kind*=user; in this case passed to the
            :class:`_DOF_estimator_user`.

        Returns
        -------
        array-like, shape (npts, 2)
            Estimation of the gradient at triangulation nodes (stored as
            degree of freedoms of reduced-HCT triangle elements).
        """
    if kind == 'user':
        if dz is None:
            raise ValueError("For a CubicTriInterpolator with *kind*='user', a valid *dz* argument is expected.")
        TE = _DOF_estimator_user(self, dz=dz)
    elif kind == 'geom':
        TE = _DOF_estimator_geom(self)
    else:
        TE = _DOF_estimator_min_E(self)
    return TE.compute_dof_from_df()