import numpy as np
def transf_constraints(constraints):
    """use QR to get transformation matrix to impose constraint

    Parameters
    ----------
    constraints : ndarray, 2-D
        restriction matrix with one constraints in rows

    Returns
    -------
    transf : ndarray
        transformation matrix to reparameterize so that constraint is
        imposed

    Notes
    -----
    This is currently and internal helper function for GAM.
    API not stable and will most likely change.

    The code for this function was taken from patsy spline handling, and
    corresponds to the reparameterization used by Wood in R's mgcv package.

    See Also
    --------
    statsmodels.base._constraints.TransformRestriction : class to impose
        constraints by reparameterization used by `_fit_constrained`.
    """
    from scipy import linalg
    m = constraints.shape[0]
    q, _ = linalg.qr(np.transpose(constraints))
    transf = q[:, m:]
    return transf