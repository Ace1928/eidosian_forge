import math
import textwrap
from abc import ABC, abstractmethod
import numpy as np
from scipy import spatial
from .._shared.utils import safe_as_int
from .._shared.compat import NP_COPY_IF_NEEDED
class FundamentalMatrixTransform(_GeometricTransform):
    """Fundamental matrix transformation.

    The fundamental matrix relates corresponding points between a pair of
    uncalibrated images. The matrix transforms homogeneous image points in one
    image to epipolar lines in the other image.

    The fundamental matrix is only defined for a pair of moving images. In the
    case of pure rotation or planar scenes, the homography describes the
    geometric relation between two images (`ProjectiveTransform`). If the
    intrinsic calibration of the images is known, the essential matrix describes
    the metric relation between the two images (`EssentialMatrixTransform`).

    References
    ----------
    .. [1] Hartley, Richard, and Andrew Zisserman. Multiple view geometry in
           computer vision. Cambridge university press, 2003.

    Parameters
    ----------
    matrix : (3, 3) array_like, optional
        Fundamental matrix.

    Attributes
    ----------
    params : (3, 3) array
        Fundamental matrix.

    Examples
    --------
    >>> import numpy as np
    >>> import skimage as ski
    >>> tform_matrix = ski.transform.FundamentalMatrixTransform()

    Define source and destination points:

    >>> src = np.array([1.839035, 1.924743,
    ...                 0.543582, 0.375221,
    ...                 0.473240, 0.142522,
    ...                 0.964910, 0.598376,
    ...                 0.102388, 0.140092,
    ...                15.994343, 9.622164,
    ...                 0.285901, 0.430055,
    ...                 0.091150, 0.254594]).reshape(-1, 2)
    >>> dst = np.array([1.002114, 1.129644,
    ...                 1.521742, 1.846002,
    ...                 1.084332, 0.275134,
    ...                 0.293328, 0.588992,
    ...                 0.839509, 0.087290,
    ...                 1.779735, 1.116857,
    ...                 0.878616, 0.602447,
    ...                 0.642616, 1.028681]).reshape(-1, 2)

    Estimate the transformation matrix:

    >>> tform_matrix.estimate(src, dst)
    True
    >>> tform_matrix.params
    array([[-0.21785884,  0.41928191, -0.03430748],
           [-0.07179414,  0.04516432,  0.02160726],
           [ 0.24806211, -0.42947814,  0.02210191]])

    Compute the Sampson distance:

    >>> tform_matrix.residuals(src, dst)
    array([0.0053886 , 0.00526101, 0.08689701, 0.01850534, 0.09418259,
           0.00185967, 0.06160489, 0.02655136])

    Apply inverse transformation:

    >>> tform_matrix.inverse(dst)
    array([[-0.0513591 ,  0.04170974,  0.01213043],
           [-0.21599496,  0.29193419,  0.00978184],
           [-0.0079222 ,  0.03758889, -0.00915389],
           [ 0.14187184, -0.27988959,  0.02476507],
           [ 0.05890075, -0.07354481, -0.00481342],
           [-0.21985267,  0.36717464, -0.01482408],
           [ 0.01339569, -0.03388123,  0.00497605],
           [ 0.03420927, -0.1135812 ,  0.02228236]])

    """

    def __init__(self, matrix=None, *, dimensionality=2):
        if matrix is None:
            matrix = np.eye(dimensionality + 1)
        else:
            matrix = np.asarray(matrix)
            dimensionality = matrix.shape[0] - 1
            if matrix.shape != (dimensionality + 1, dimensionality + 1):
                raise ValueError('Invalid shape of transformation matrix')
        self.params = matrix
        if dimensionality != 2:
            raise NotImplementedError(f'{self.__class__} is only implemented for 2D coordinates (i.e. 3D transformation matrices).')

    def __call__(self, coords):
        """Apply forward transformation.

        Parameters
        ----------
        coords : (N, 2) array_like
            Source coordinates.

        Returns
        -------
        coords : (N, 3) array
            Epipolar lines in the destination image.

        """
        coords = np.asarray(coords)
        coords_homogeneous = np.column_stack([coords, np.ones(coords.shape[0])])
        return coords_homogeneous @ self.params.T

    @property
    def inverse(self):
        """Return a transform object representing the inverse.

        See Hartley & Zisserman, Ch. 8: Epipolar Geometry and the Fundamental
        Matrix, for an explanation of why F.T gives the inverse.
        """
        return type(self)(matrix=self.params.T)

    def _setup_constraint_matrix(self, src, dst):
        """Setup and solve the homogeneous epipolar constraint matrix::

            dst' * F * src = 0.

        Parameters
        ----------
        src : (N, 2) array_like
            Source coordinates.
        dst : (N, 2) array_like
            Destination coordinates.

        Returns
        -------
        F_normalized : (3, 3) array
            The normalized solution to the homogeneous system. If the system
            is not well-conditioned, this matrix contains NaNs.
        src_matrix : (3, 3) array
            The transformation matrix to obtain the normalized source
            coordinates.
        dst_matrix : (3, 3) array
            The transformation matrix to obtain the normalized destination
            coordinates.

        """
        src = np.asarray(src)
        dst = np.asarray(dst)
        if src.shape != dst.shape:
            raise ValueError('src and dst shapes must be identical.')
        if src.shape[0] < 8:
            raise ValueError('src.shape[0] must be equal or larger than 8.')
        try:
            src_matrix, src = _center_and_normalize_points(src)
            dst_matrix, dst = _center_and_normalize_points(dst)
        except ZeroDivisionError:
            self.params = np.full((3, 3), np.nan)
            return 3 * [np.full((3, 3), np.nan)]
        A = np.ones((src.shape[0], 9))
        A[:, :2] = src
        A[:, :3] *= dst[:, 0, np.newaxis]
        A[:, 3:5] = src
        A[:, 3:6] *= dst[:, 1, np.newaxis]
        A[:, 6:8] = src
        _, _, V = np.linalg.svd(A)
        F_normalized = V[-1, :].reshape(3, 3)
        return (F_normalized, src_matrix, dst_matrix)

    def estimate(self, src, dst):
        """Estimate fundamental matrix using 8-point algorithm.

        The 8-point algorithm requires at least 8 corresponding point pairs for
        a well-conditioned solution, otherwise the over-determined solution is
        estimated.

        Parameters
        ----------
        src : (N, 2) array_like
            Source coordinates.
        dst : (N, 2) array_like
            Destination coordinates.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """
        F_normalized, src_matrix, dst_matrix = self._setup_constraint_matrix(src, dst)
        U, S, V = np.linalg.svd(F_normalized)
        S[2] = 0
        F = U @ np.diag(S) @ V
        self.params = dst_matrix.T @ F @ src_matrix
        return True

    def residuals(self, src, dst):
        """Compute the Sampson distance.

        The Sampson distance is the first approximation to the geometric error.

        Parameters
        ----------
        src : (N, 2) array
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.

        Returns
        -------
        residuals : (N,) array
            Sampson distance.

        """
        src_homogeneous = np.column_stack([src, np.ones(src.shape[0])])
        dst_homogeneous = np.column_stack([dst, np.ones(dst.shape[0])])
        F_src = self.params @ src_homogeneous.T
        Ft_dst = self.params.T @ dst_homogeneous.T
        dst_F_src = np.sum(dst_homogeneous * F_src.T, axis=1)
        return np.abs(dst_F_src) / np.sqrt(F_src[0] ** 2 + F_src[1] ** 2 + Ft_dst[0] ** 2 + Ft_dst[1] ** 2)