from abc import ABCMeta, abstractmethod
from typing import List
import numpy as np
from joblib import effective_n_jobs
from ..base import BaseEstimator, MetaEstimatorMixin, clone, is_classifier, is_regressor
from ..utils import Bunch, _print_elapsed_time, check_random_state
from ..utils._tags import _safe_tags
from ..utils.metaestimators import _BaseComposition
class _BaseHeterogeneousEnsemble(MetaEstimatorMixin, _BaseComposition, metaclass=ABCMeta):
    """Base class for heterogeneous ensemble of learners.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        The ensemble of estimators to use in the ensemble. Each element of the
        list is defined as a tuple of string (i.e. name of the estimator) and
        an estimator instance. An estimator can be set to `'drop'` using
        `set_params`.

    Attributes
    ----------
    estimators_ : list of estimators
        The elements of the estimators parameter, having been fitted on the
        training data. If an estimator has been set to `'drop'`, it will not
        appear in `estimators_`.
    """
    _required_parameters = ['estimators']

    @property
    def named_estimators(self):
        """Dictionary to access any fitted sub-estimators by name.

        Returns
        -------
        :class:`~sklearn.utils.Bunch`
        """
        return Bunch(**dict(self.estimators))

    @abstractmethod
    def __init__(self, estimators):
        self.estimators = estimators

    def _validate_estimators(self):
        if len(self.estimators) == 0:
            raise ValueError("Invalid 'estimators' attribute, 'estimators' should be a non-empty list of (string, estimator) tuples.")
        names, estimators = zip(*self.estimators)
        self._validate_names(names)
        has_estimator = any((est != 'drop' for est in estimators))
        if not has_estimator:
            raise ValueError('All estimators are dropped. At least one is required to be an estimator.')
        is_estimator_type = is_classifier if is_classifier(self) else is_regressor
        for est in estimators:
            if est != 'drop' and (not is_estimator_type(est)):
                raise ValueError('The estimator {} should be a {}.'.format(est.__class__.__name__, is_estimator_type.__name__[3:]))
        return (names, estimators)

    def set_params(self, **params):
        """
        Set the parameters of an estimator from the ensemble.

        Valid parameter keys can be listed with `get_params()`. Note that you
        can directly set the parameters of the estimators contained in
        `estimators`.

        Parameters
        ----------
        **params : keyword arguments
            Specific parameters using e.g.
            `set_params(parameter_name=new_value)`. In addition, to setting the
            parameters of the estimator, the individual estimator of the
            estimators can also be set, or can be removed by setting them to
            'drop'.

        Returns
        -------
        self : object
            Estimator instance.
        """
        super()._set_params('estimators', **params)
        return self

    def get_params(self, deep=True):
        """
        Get the parameters of an estimator from the ensemble.

        Returns the parameters given in the constructor as well as the
        estimators contained within the `estimators` parameter.

        Parameters
        ----------
        deep : bool, default=True
            Setting it to True gets the various estimators and the parameters
            of the estimators as well.

        Returns
        -------
        params : dict
            Parameter and estimator names mapped to their values or parameter
            names mapped to their values.
        """
        return super()._get_params('estimators', deep=deep)

    def _more_tags(self):
        try:
            allow_nan = all((_safe_tags(est[1])['allow_nan'] if est[1] != 'drop' else True for est in self.estimators))
        except Exception:
            allow_nan = False
        return {'preserves_dtype': [], 'allow_nan': allow_nan}