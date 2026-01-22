import warnings
import numpy
from numpy import (array, isfinite, inexact, nonzero, iscomplexobj,
from scipy._lib._util import _asarray_validated
from ._misc import LinAlgError, _datacopied, norm
from .lapack import get_lapack_funcs, _compute_lwork
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args

    Converts complex eigenvalues ``w`` and eigenvectors ``v`` to real
    eigenvalues in a block diagonal form ``wr`` and the associated real
    eigenvectors ``vr``, such that::

        vr @ wr = X @ vr

    continues to hold, where ``X`` is the original array for which ``w`` and
    ``v`` are the eigenvalues and eigenvectors.

    .. versionadded:: 1.1.0

    Parameters
    ----------
    w : (..., M) array_like
        Complex or real eigenvalues, an array or stack of arrays

        Conjugate pairs must not be interleaved, else the wrong result
        will be produced. So ``[1+1j, 1, 1-1j]`` will give a correct result,
        but ``[1+1j, 2+1j, 1-1j, 2-1j]`` will not.

    v : (..., M, M) array_like
        Complex or real eigenvectors, a square array or stack of square arrays.

    Returns
    -------
    wr : (..., M, M) ndarray
        Real diagonal block form of eigenvalues
    vr : (..., M, M) ndarray
        Real eigenvectors associated with ``wr``

    See Also
    --------
    eig : Eigenvalues and right eigenvectors for non-symmetric arrays
    rsf2csf : Convert real Schur form to complex Schur form

    Notes
    -----
    ``w``, ``v`` must be the eigenstructure for some *real* matrix ``X``.
    For example, obtained by ``w, v = scipy.linalg.eig(X)`` or
    ``w, v = numpy.linalg.eig(X)`` in which case ``X`` can also represent
    stacked arrays.

    .. versionadded:: 1.1.0

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2, 3], [0, 4, 5], [0, -5, 4]])
    >>> X
    array([[ 1,  2,  3],
           [ 0,  4,  5],
           [ 0, -5,  4]])

    >>> from scipy import linalg
    >>> w, v = linalg.eig(X)
    >>> w
    array([ 1.+0.j,  4.+5.j,  4.-5.j])
    >>> v
    array([[ 1.00000+0.j     , -0.01906-0.40016j, -0.01906+0.40016j],
           [ 0.00000+0.j     ,  0.00000-0.64788j,  0.00000+0.64788j],
           [ 0.00000+0.j     ,  0.64788+0.j     ,  0.64788-0.j     ]])

    >>> wr, vr = linalg.cdf2rdf(w, v)
    >>> wr
    array([[ 1.,  0.,  0.],
           [ 0.,  4.,  5.],
           [ 0., -5.,  4.]])
    >>> vr
    array([[ 1.     ,  0.40016, -0.01906],
           [ 0.     ,  0.64788,  0.     ],
           [ 0.     ,  0.     ,  0.64788]])

    >>> vr @ wr
    array([[ 1.     ,  1.69593,  1.9246 ],
           [ 0.     ,  2.59153,  3.23942],
           [ 0.     , -3.23942,  2.59153]])
    >>> X @ vr
    array([[ 1.     ,  1.69593,  1.9246 ],
           [ 0.     ,  2.59153,  3.23942],
           [ 0.     , -3.23942,  2.59153]])
    