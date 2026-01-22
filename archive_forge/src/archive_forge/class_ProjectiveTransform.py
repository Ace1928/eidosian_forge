import math
import textwrap
from abc import ABC, abstractmethod
import numpy as np
from scipy import spatial
from .._shared.utils import safe_as_int
from .._shared.compat import NP_COPY_IF_NEEDED
class ProjectiveTransform(_GeometricTransform):
    """Projective transformation.

    Apply a projective transformation (homography) on coordinates.

    For each homogeneous coordinate :math:`\\mathbf{x} = [x, y, 1]^T`, its
    target position is calculated by multiplying with the given matrix,
    :math:`H`, to give :math:`H \\mathbf{x}`::

      [[a0 a1 a2]
       [b0 b1 b2]
       [c0 c1 1 ]].

    E.g., to rotate by theta degrees clockwise, the matrix should be::

      [[cos(theta) -sin(theta) 0]
       [sin(theta)  cos(theta) 0]
       [0            0         1]]

    or, to translate x by 10 and y by 20::

      [[1 0 10]
       [0 1 20]
       [0 0 1 ]].

    Parameters
    ----------
    matrix : (D+1, D+1) array_like, optional
        Homogeneous transformation matrix.
    dimensionality : int, optional
        The number of dimensions of the transform. This is ignored if
        ``matrix`` is not None.

    Attributes
    ----------
    params : (D+1, D+1) array
        Homogeneous transformation matrix.

    """

    def __init__(self, matrix=None, *, dimensionality=2):
        if matrix is None:
            matrix = np.eye(dimensionality + 1)
        else:
            matrix = np.asarray(matrix)
            dimensionality = matrix.shape[0] - 1
            if matrix.shape != (dimensionality + 1, dimensionality + 1):
                raise ValueError('invalid shape of transformation matrix')
        self.params = matrix
        self._coeffs = range(matrix.size - 1)

    @property
    def _inv_matrix(self):
        return np.linalg.inv(self.params)

    def _apply_mat(self, coords, matrix):
        ndim = matrix.shape[0] - 1
        coords = np.array(coords, copy=NP_COPY_IF_NEEDED, ndmin=2)
        src = np.concatenate([coords, np.ones((coords.shape[0], 1))], axis=1)
        dst = src @ matrix.T
        dst[dst[:, ndim] == 0, ndim] = np.finfo(float).eps
        dst[:, :ndim] /= dst[:, ndim:ndim + 1]
        return dst[:, :ndim]

    def __array__(self, dtype=None):
        if dtype is None:
            return self.params
        else:
            return self.params.astype(dtype)

    def __call__(self, coords):
        """Apply forward transformation.

        Parameters
        ----------
        coords : (N, D) array_like
            Source coordinates.

        Returns
        -------
        coords_out : (N, D) array
            Destination coordinates.

        """
        return self._apply_mat(coords, self.params)

    @property
    def inverse(self):
        """Return a transform object representing the inverse."""
        return type(self)(matrix=self._inv_matrix)

    def estimate(self, src, dst, weights=None):
        """Estimate the transformation from a set of corresponding points.

        You can determine the over-, well- and under-determined parameters
        with the total least-squares method.

        Number of source and destination coordinates must match.

        The transformation is defined as::

            X = (a0*x + a1*y + a2) / (c0*x + c1*y + 1)
            Y = (b0*x + b1*y + b2) / (c0*x + c1*y + 1)

        These equations can be transformed to the following form::

            0 = a0*x + a1*y + a2 - c0*x*X - c1*y*X - X
            0 = b0*x + b1*y + b2 - c0*x*Y - c1*y*Y - Y

        which exist for each set of corresponding points, so we have a set of
        N * 2 equations. The coefficients appear linearly so we can write
        A x = 0, where::

            A   = [[x y 1 0 0 0 -x*X -y*X -X]
                   [0 0 0 x y 1 -x*Y -y*Y -Y]
                    ...
                    ...
                  ]
            x.T = [a0 a1 a2 b0 b1 b2 c0 c1 c3]

        In case of total least-squares the solution of this homogeneous system
        of equations is the right singular vector of A which corresponds to the
        smallest singular value normed by the coefficient c3.

        Weights can be applied to each pair of corresponding points to
        indicate, particularly in an overdetermined system, if point pairs have
        higher or lower confidence or uncertainties associated with them. From
        the matrix treatment of least squares problems, these weight values are
        normalised, square-rooted, then built into a diagonal matrix, by which
        A is multiplied.

        In case of the affine transformation the coefficients c0 and c1 are 0.
        Thus the system of equations is::

            A   = [[x y 1 0 0 0 -X]
                   [0 0 0 x y 1 -Y]
                    ...
                    ...
                  ]
            x.T = [a0 a1 a2 b0 b1 b2 c3]

        Parameters
        ----------
        src : (N, 2) array_like
            Source coordinates.
        dst : (N, 2) array_like
            Destination coordinates.
        weights : (N,) array_like, optional
            Relative weight values for each pair of points.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """
        src = np.asarray(src)
        dst = np.asarray(dst)
        n, d = src.shape
        src_matrix, src = _center_and_normalize_points(src)
        dst_matrix, dst = _center_and_normalize_points(dst)
        if not np.all(np.isfinite(src_matrix + dst_matrix)):
            self.params = np.full((d + 1, d + 1), np.nan)
            return False
        A = np.zeros((n * d, (d + 1) ** 2))
        for ddim in range(d):
            A[ddim * n:(ddim + 1) * n, ddim * (d + 1):ddim * (d + 1) + d] = src
            A[ddim * n:(ddim + 1) * n, ddim * (d + 1) + d] = 1
            A[ddim * n:(ddim + 1) * n, -d - 1:-1] = src
            A[ddim * n:(ddim + 1) * n, -1] = -1
            A[ddim * n:(ddim + 1) * n, -d - 1:] *= -dst[:, ddim:ddim + 1]
        A = A[:, list(self._coeffs) + [-1]]
        if weights is None:
            _, _, V = np.linalg.svd(A)
        else:
            weights = np.asarray(weights)
            W = np.diag(np.tile(np.sqrt(weights / np.max(weights)), d))
            _, _, V = np.linalg.svd(W @ A)
        if np.isclose(V[-1, -1], 0):
            self.params = np.full((d + 1, d + 1), np.nan)
            return False
        H = np.zeros((d + 1, d + 1))
        H.flat[list(self._coeffs) + [-1]] = -V[-1, :-1] / V[-1, -1]
        H[d, d] = 1
        H = np.linalg.inv(dst_matrix) @ H @ src_matrix
        H /= H[-1, -1]
        self.params = H
        return True

    def __add__(self, other):
        """Combine this transformation with another."""
        if isinstance(other, ProjectiveTransform):
            if type(self) == type(other):
                tform = self.__class__
            else:
                tform = ProjectiveTransform
            return tform(other.params @ self.params)
        else:
            raise TypeError('Cannot combine transformations of differing types.')

    def __nice__(self):
        """common 'paramstr' used by __str__ and __repr__"""
        npstring = np.array2string(self.params, separator=', ')
        paramstr = 'matrix=\n' + textwrap.indent(npstring, '    ')
        return paramstr

    def __repr__(self):
        """Add standard repr formatting around a __nice__ string"""
        paramstr = self.__nice__()
        classname = self.__class__.__name__
        classstr = classname
        return f'<{classstr}({paramstr}) at {hex(id(self))}>'

    def __str__(self):
        """Add standard str formatting around a __nice__ string"""
        paramstr = self.__nice__()
        classname = self.__class__.__name__
        classstr = classname
        return f'<{classstr}({paramstr})>'

    @property
    def dimensionality(self):
        """The dimensionality of the transformation."""
        return self.params.shape[0] - 1