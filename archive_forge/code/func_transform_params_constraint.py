import numpy as np
def transform_params_constraint(params, Sinv, R, q):
    """find the parameters that statisfy linear constraint from unconstrained

    The linear constraint R params = q is imposed.

    Parameters
    ----------
    params : array_like
        unconstrained parameters
    Sinv : ndarray, 2d, symmetric
        covariance matrix of the parameter estimate
    R : ndarray, 2d
        constraint matrix
    q : ndarray, 1d
        values of the constraint

    Returns
    -------
    params_constraint : ndarray
        parameters of the same length as params satisfying the constraint

    Notes
    -----
    This is the exact formula for OLS and other linear models. It will be
    a local approximation for nonlinear models.

    TODO: Is Sinv always the covariance matrix?
    In the linear case it can be (X'X)^{-1} or sigmahat^2 (X'X)^{-1}.

    My guess is that this is the point in the subspace that satisfies
    the constraint that has minimum Mahalanobis distance. Proof ?
    """
    rsr = R.dot(Sinv).dot(R.T)
    reduction = Sinv.dot(R.T).dot(np.linalg.solve(rsr, R.dot(params) - q))
    return params - reduction