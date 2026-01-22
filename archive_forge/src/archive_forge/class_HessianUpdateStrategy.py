import numpy as np
from numpy.linalg import norm
from scipy.linalg import get_blas_funcs
from warnings import warn
class HessianUpdateStrategy:
    """Interface for implementing Hessian update strategies.

    Many optimization methods make use of Hessian (or inverse Hessian)
    approximations, such as the quasi-Newton methods BFGS, SR1, L-BFGS.
    Some of these  approximations, however, do not actually need to store
    the entire matrix or can compute the internal matrix product with a
    given vector in a very efficiently manner. This class serves as an
    abstract interface between the optimization algorithm and the
    quasi-Newton update strategies, giving freedom of implementation
    to store and update the internal matrix as efficiently as possible.
    Different choices of initialization and update procedure will result
    in different quasi-Newton strategies.

    Four methods should be implemented in derived classes: ``initialize``,
    ``update``, ``dot`` and ``get_matrix``.

    Notes
    -----
    Any instance of a class that implements this interface,
    can be accepted by the method ``minimize`` and used by
    the compatible solvers to approximate the Hessian (or
    inverse Hessian) used by the optimization algorithms.
    """

    def initialize(self, n, approx_type):
        """Initialize internal matrix.

        Allocate internal memory for storing and updating
        the Hessian or its inverse.

        Parameters
        ----------
        n : int
            Problem dimension.
        approx_type : {'hess', 'inv_hess'}
            Selects either the Hessian or the inverse Hessian.
            When set to 'hess' the Hessian will be stored and updated.
            When set to 'inv_hess' its inverse will be used instead.
        """
        raise NotImplementedError('The method ``initialize(n, approx_type)`` is not implemented.')

    def update(self, delta_x, delta_grad):
        """Update internal matrix.

        Update Hessian matrix or its inverse (depending on how 'approx_type'
        is defined) using information about the last evaluated points.

        Parameters
        ----------
        delta_x : ndarray
            The difference between two points the gradient
            function have been evaluated at: ``delta_x = x2 - x1``.
        delta_grad : ndarray
            The difference between the gradients:
            ``delta_grad = grad(x2) - grad(x1)``.
        """
        raise NotImplementedError('The method ``update(delta_x, delta_grad)`` is not implemented.')

    def dot(self, p):
        """Compute the product of the internal matrix with the given vector.

        Parameters
        ----------
        p : array_like
            1-D array representing a vector.

        Returns
        -------
        Hp : array
            1-D represents the result of multiplying the approximation matrix
            by vector p.
        """
        raise NotImplementedError('The method ``dot(p)`` is not implemented.')

    def get_matrix(self):
        """Return current internal matrix.

        Returns
        -------
        H : ndarray, shape (n, n)
            Dense matrix containing either the Hessian
            or its inverse (depending on how 'approx_type'
            is defined).
        """
        raise NotImplementedError('The method ``get_matrix(p)`` is not implemented.')