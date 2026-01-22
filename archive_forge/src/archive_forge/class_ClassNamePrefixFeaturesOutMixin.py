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
class ClassNamePrefixFeaturesOutMixin:
    """Mixin class for transformers that generate their own names by prefixing.

    This mixin is useful when the transformer needs to generate its own feature
    names out, such as :class:`~sklearn.decomposition.PCA`. For example, if
    :class:`~sklearn.decomposition.PCA` outputs 3 features, then the generated feature
    names out are: `["pca0", "pca1", "pca2"]`.

    This mixin assumes that a `_n_features_out` attribute is defined when the
    transformer is fitted. `_n_features_out` is the number of output features
    that the transformer will return in `transform` of `fit_transform`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import ClassNamePrefixFeaturesOutMixin
    >>> class MyEstimator(ClassNamePrefixFeaturesOutMixin):
    ...     def fit(self, X, y=None):
    ...         self._n_features_out = X.shape[1]
    ...         return self
    >>> X = np.array([[1, 2], [3, 4]])
    >>> MyEstimator().fit(X).get_feature_names_out()
    array(['myestimator0', 'myestimator1'], dtype=object)
    """

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        The feature names out will prefixed by the lowercased class name. For
        example, if the transformer outputs 3 features, then the feature names
        out are: `["class_name0", "class_name1", "class_name2"]`.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Only used to validate feature names with the names seen in `fit`.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        check_is_fitted(self, '_n_features_out')
        return _generate_get_feature_names_out(self, self._n_features_out, input_features=input_features)