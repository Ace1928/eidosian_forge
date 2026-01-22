import collections
import inspect
import logging
import pkgutil
import platform
import warnings
from copy import deepcopy
from importlib import import_module
from numbers import Number
from operator import itemgetter
import numpy as np
from packaging.version import Version
from mlflow import MlflowClient
from mlflow.utils.arguments_utils import _get_arg_names
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from mlflow.utils.time import get_current_time_millis
def _backported_all_estimators(type_filter=None):
    """
    Backported from scikit-learn 0.23.2:
    https://github.com/scikit-learn/scikit-learn/blob/0.23.2/sklearn/utils/__init__.py#L1146

    Use this backported `all_estimators` in old versions of sklearn because:
    1. An inferior version of `all_estimators` that old versions of sklearn use for testing,
       might function differently from a newer version.
    2. This backported `all_estimators` works on old versions of sklearn that don’t even define
       the testing utility variant of `all_estimators`.

    ========== original docstring ==========
    Get a list of all estimators from sklearn.
    This function crawls the module and gets all classes that inherit
    from BaseEstimator. Classes that are defined in test-modules are not
    included.
    By default meta_estimators such as GridSearchCV are also not included.
    Parameters
    ----------
    type_filter : string, list of string,  or None, default=None
        Which kind of estimators should be returned. If None, no filter is
        applied and all estimators are returned.  Possible values are
        'classifier', 'regressor', 'cluster' and 'transformer' to get
        estimators only of these specific types, or a list of these to
        get the estimators that fit at least one of the types.

    Returns
    -------
    estimators : list of tuples
        List of (name, class), where ``name`` is the class name as string
        and ``class`` is the actual type of the class.
    """
    import sklearn
    from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin, RegressorMixin, TransformerMixin
    from sklearn.utils._testing import ignore_warnings
    IS_PYPY = platform.python_implementation() == 'PyPy'

    def is_abstract(c):
        if not hasattr(c, '__abstractmethods__'):
            return False
        if not len(c.__abstractmethods__):
            return False
        return True
    all_classes = []
    modules_to_ignore = {'tests', 'externals', 'setup', 'conftest'}
    root = sklearn.__path__[0]
    with ignore_warnings(category=FutureWarning):
        for _, modname, _ in pkgutil.walk_packages(path=[root], prefix='sklearn.'):
            mod_parts = modname.split('.')
            if any((part in modules_to_ignore for part in mod_parts)) or '._' in modname:
                continue
            module = import_module(modname)
            classes = inspect.getmembers(module, inspect.isclass)
            classes = [(name, est_cls) for name, est_cls in classes if not name.startswith('_')]
            if IS_PYPY and 'feature_extraction' in modname:
                classes = [(name, est_cls) for name, est_cls in classes if name == 'FeatureHasher']
            all_classes.extend(classes)
    all_classes = set(all_classes)
    estimators = [c for c in all_classes if issubclass(c[1], BaseEstimator) and c[0] != 'BaseEstimator']
    estimators = [c for c in estimators if not is_abstract(c[1])]
    if type_filter is not None:
        type_filter = list(type_filter) if isinstance(type_filter, list) else [type_filter]
        filtered_estimators = []
        filters = {'classifier': ClassifierMixin, 'regressor': RegressorMixin, 'transformer': TransformerMixin, 'cluster': ClusterMixin}
        for name, mixin in filters.items():
            if name in type_filter:
                type_filter.remove(name)
                filtered_estimators.extend([est for est in estimators if issubclass(est[1], mixin)])
        estimators = filtered_estimators
        if type_filter:
            raise ValueError("Parameter type_filter must be 'classifier', 'regressor', 'transformer', 'cluster' or None, got %s." % repr(type_filter))
    return sorted(set(estimators), key=itemgetter(0))