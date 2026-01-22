import numbers
import operator
import sys
import warnings
from contextlib import suppress
from functools import reduce, wraps
from inspect import Parameter, isclass, signature
import joblib
import numpy as np
import scipy.sparse as sp
from .. import get_config as _get_config
from ..exceptions import DataConversionWarning, NotFittedError, PositiveSpectrumWarning
from ..utils._array_api import _asarray_with_order, _is_numpy_namespace, get_namespace
from ..utils.fixes import ComplexWarning, _preserve_dia_indices_dtype
from ._isfinite import FiniteStatus, cy_isfinite
from .fixes import _object_dtype_isnan
def _check_psd_eigenvalues(lambdas, enable_warnings=False):
    """Check the eigenvalues of a positive semidefinite (PSD) matrix.

    Checks the provided array of PSD matrix eigenvalues for numerical or
    conditioning issues and returns a fixed validated version. This method
    should typically be used if the PSD matrix is user-provided (e.g. a
    Gram matrix) or computed using a user-provided dissimilarity metric
    (e.g. kernel function), or if the decomposition process uses approximation
    methods (randomized SVD, etc.).

    It checks for three things:

    - that there are no significant imaginary parts in eigenvalues (more than
      1e-5 times the maximum real part). If this check fails, it raises a
      ``ValueError``. Otherwise all non-significant imaginary parts that may
      remain are set to zero. This operation is traced with a
      ``PositiveSpectrumWarning`` when ``enable_warnings=True``.

    - that eigenvalues are not all negative. If this check fails, it raises a
      ``ValueError``

    - that there are no significant negative eigenvalues with absolute value
      more than 1e-10 (1e-6) and more than 1e-5 (5e-3) times the largest
      positive eigenvalue in double (simple) precision. If this check fails,
      it raises a ``ValueError``. Otherwise all negative eigenvalues that may
      remain are set to zero. This operation is traced with a
      ``PositiveSpectrumWarning`` when ``enable_warnings=True``.

    Finally, all the positive eigenvalues that are too small (with a value
    smaller than the maximum eigenvalue multiplied by 1e-12 (2e-7)) are set to
    zero. This operation is traced with a ``PositiveSpectrumWarning`` when
    ``enable_warnings=True``.

    Parameters
    ----------
    lambdas : array-like of shape (n_eigenvalues,)
        Array of eigenvalues to check / fix.

    enable_warnings : bool, default=False
        When this is set to ``True``, a ``PositiveSpectrumWarning`` will be
        raised when there are imaginary parts, negative eigenvalues, or
        extremely small non-zero eigenvalues. Otherwise no warning will be
        raised. In both cases, imaginary parts, negative eigenvalues, and
        extremely small non-zero eigenvalues will be set to zero.

    Returns
    -------
    lambdas_fixed : ndarray of shape (n_eigenvalues,)
        A fixed validated copy of the array of eigenvalues.

    Examples
    --------
    >>> from sklearn.utils.validation import _check_psd_eigenvalues
    >>> _check_psd_eigenvalues([1, 2])      # nominal case
    array([1, 2])
    >>> _check_psd_eigenvalues([5, 5j])     # significant imag part
    Traceback (most recent call last):
        ...
    ValueError: There are significant imaginary parts in eigenvalues (1
        of the maximum real part). Either the matrix is not PSD, or there was
        an issue while computing the eigendecomposition of the matrix.
    >>> _check_psd_eigenvalues([5, 5e-5j])  # insignificant imag part
    array([5., 0.])
    >>> _check_psd_eigenvalues([-5, -1])    # all negative
    Traceback (most recent call last):
        ...
    ValueError: All eigenvalues are negative (maximum is -1). Either the
        matrix is not PSD, or there was an issue while computing the
        eigendecomposition of the matrix.
    >>> _check_psd_eigenvalues([5, -1])     # significant negative
    Traceback (most recent call last):
        ...
    ValueError: There are significant negative eigenvalues (0.2 of the
        maximum positive). Either the matrix is not PSD, or there was an issue
        while computing the eigendecomposition of the matrix.
    >>> _check_psd_eigenvalues([5, -5e-5])  # insignificant negative
    array([5., 0.])
    >>> _check_psd_eigenvalues([5, 4e-12])  # bad conditioning (too small)
    array([5., 0.])

    """
    lambdas = np.array(lambdas)
    is_double_precision = lambdas.dtype == np.float64
    significant_imag_ratio = 1e-05
    significant_neg_ratio = 1e-05 if is_double_precision else 0.005
    significant_neg_value = 1e-10 if is_double_precision else 1e-06
    small_pos_ratio = 1e-12 if is_double_precision else 2e-07
    if not np.isreal(lambdas).all():
        max_imag_abs = np.abs(np.imag(lambdas)).max()
        max_real_abs = np.abs(np.real(lambdas)).max()
        if max_imag_abs > significant_imag_ratio * max_real_abs:
            raise ValueError('There are significant imaginary parts in eigenvalues (%g of the maximum real part). Either the matrix is not PSD, or there was an issue while computing the eigendecomposition of the matrix.' % (max_imag_abs / max_real_abs))
        if enable_warnings:
            warnings.warn('There are imaginary parts in eigenvalues (%g of the maximum real part). Either the matrix is not PSD, or there was an issue while computing the eigendecomposition of the matrix. Only the real parts will be kept.' % (max_imag_abs / max_real_abs), PositiveSpectrumWarning)
    lambdas = np.real(lambdas)
    max_eig = lambdas.max()
    if max_eig < 0:
        raise ValueError('All eigenvalues are negative (maximum is %g). Either the matrix is not PSD, or there was an issue while computing the eigendecomposition of the matrix.' % max_eig)
    else:
        min_eig = lambdas.min()
        if min_eig < -significant_neg_ratio * max_eig and min_eig < -significant_neg_value:
            raise ValueError('There are significant negative eigenvalues (%g of the maximum positive). Either the matrix is not PSD, or there was an issue while computing the eigendecomposition of the matrix.' % (-min_eig / max_eig))
        elif min_eig < 0:
            if enable_warnings:
                warnings.warn('There are negative eigenvalues (%g of the maximum positive). Either the matrix is not PSD, or there was an issue while computing the eigendecomposition of the matrix. Negative eigenvalues will be replaced with 0.' % (-min_eig / max_eig), PositiveSpectrumWarning)
            lambdas[lambdas < 0] = 0
    too_small_lambdas = (0 < lambdas) & (lambdas < small_pos_ratio * max_eig)
    if too_small_lambdas.any():
        if enable_warnings:
            warnings.warn('Badly conditioned PSD matrix spectrum: the largest eigenvalue is more than %g times the smallest. Small eigenvalues will be replaced with 0.' % (1 / small_pos_ratio), PositiveSpectrumWarning)
        lambdas[too_small_lambdas] = 0
    return lambdas