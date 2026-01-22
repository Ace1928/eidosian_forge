import copy
import functools
import inspect
import platform
import re
import warnings
from collections import defaultdict
import numpy as np
from . import __version__
from ._config import config_context, get_config
from .exceptions import InconsistentVersionWarning
from .utils import _IS_32BIT
from .utils._estimator_html_repr import _HTMLDocumentationLinkMixin, estimator_html_repr
from .utils._metadata_requests import _MetadataRequester, _routing_enabled
from .utils._param_validation import validate_parameter_constraints
from .utils._set_output import _SetOutputMixin
from .utils._tags import (
from .utils.validation import (
class BiclusterMixin:
    """Mixin class for all bicluster estimators in scikit-learn.

    This mixin defines the following functionality:

    - `biclusters_` property that returns the row and column indicators;
    - `get_indices` method that returns the row and column indices of a bicluster;
    - `get_shape` method that returns the shape of a bicluster;
    - `get_submatrix` method that returns the submatrix corresponding to a bicluster.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import BaseEstimator, BiclusterMixin
    >>> class DummyBiClustering(BiclusterMixin, BaseEstimator):
    ...     def fit(self, X, y=None):
    ...         self.rows_ = np.ones(shape=(1, X.shape[0]), dtype=bool)
    ...         self.columns_ = np.ones(shape=(1, X.shape[1]), dtype=bool)
    ...         return self
    >>> X = np.array([[1, 1], [2, 1], [1, 0],
    ...               [4, 7], [3, 5], [3, 6]])
    >>> bicluster = DummyBiClustering().fit(X)
    >>> hasattr(bicluster, "biclusters_")
    True
    >>> bicluster.get_indices(0)
    (array([0, 1, 2, 3, 4, 5]), array([0, 1]))
    """

    @property
    def biclusters_(self):
        """Convenient way to get row and column indicators together.

        Returns the ``rows_`` and ``columns_`` members.
        """
        return (self.rows_, self.columns_)

    def get_indices(self, i):
        """Row and column indices of the `i`'th bicluster.

        Only works if ``rows_`` and ``columns_`` attributes exist.

        Parameters
        ----------
        i : int
            The index of the cluster.

        Returns
        -------
        row_ind : ndarray, dtype=np.intp
            Indices of rows in the dataset that belong to the bicluster.
        col_ind : ndarray, dtype=np.intp
            Indices of columns in the dataset that belong to the bicluster.
        """
        rows = self.rows_[i]
        columns = self.columns_[i]
        return (np.nonzero(rows)[0], np.nonzero(columns)[0])

    def get_shape(self, i):
        """Shape of the `i`'th bicluster.

        Parameters
        ----------
        i : int
            The index of the cluster.

        Returns
        -------
        n_rows : int
            Number of rows in the bicluster.

        n_cols : int
            Number of columns in the bicluster.
        """
        indices = self.get_indices(i)
        return tuple((len(i) for i in indices))

    def get_submatrix(self, i, data):
        """Return the submatrix corresponding to bicluster `i`.

        Parameters
        ----------
        i : int
            The index of the cluster.
        data : array-like of shape (n_samples, n_features)
            The data.

        Returns
        -------
        submatrix : ndarray of shape (n_rows, n_cols)
            The submatrix corresponding to bicluster `i`.

        Notes
        -----
        Works with sparse matrices. Only works if ``rows_`` and
        ``columns_`` attributes exist.
        """
        from .utils.validation import check_array
        data = check_array(data, accept_sparse='csr')
        row_ind, col_ind = self.get_indices(i)
        return data[row_ind[:, np.newaxis], col_ind]