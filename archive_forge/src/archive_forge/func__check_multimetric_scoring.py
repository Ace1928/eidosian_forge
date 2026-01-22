import copy
import warnings
from collections import Counter
from functools import partial
from inspect import signature
from traceback import format_exc
from ..base import is_regressor
from ..utils import Bunch
from ..utils._param_validation import HasMethods, Hidden, StrOptions, validate_params
from ..utils._response import _get_response_values
from ..utils.metadata_routing import (
from ..utils.validation import _check_response_method
from . import (
from .cluster import (
def _check_multimetric_scoring(estimator, scoring):
    """Check the scoring parameter in cases when multiple metrics are allowed.

    In addition, multimetric scoring leverages a caching mechanism to not call the same
    estimator response method multiple times. Hence, the scorer is modified to only use
    a single response method given a list of response methods and the estimator.

    Parameters
    ----------
    estimator : sklearn estimator instance
        The estimator for which the scoring will be applied.

    scoring : list, tuple or dict
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

        The possibilities are:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where they keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.

        See :ref:`multimetric_grid_search` for an example.

    Returns
    -------
    scorers_dict : dict
        A dict mapping each scorer name to its validated scorer.
    """
    err_msg_generic = f'scoring is invalid (got {scoring!r}). Refer to the scoring glossary for details: https://scikit-learn.org/stable/glossary.html#term-scoring'
    if isinstance(scoring, (list, tuple, set)):
        err_msg = 'The list/tuple elements must be unique strings of predefined scorers. '
        try:
            keys = set(scoring)
        except TypeError as e:
            raise ValueError(err_msg) from e
        if len(keys) != len(scoring):
            raise ValueError(f'{err_msg} Duplicate elements were found in the given list. {scoring!r}')
        elif len(keys) > 0:
            if not all((isinstance(k, str) for k in keys)):
                if any((callable(k) for k in keys)):
                    raise ValueError(f'{err_msg} One or more of the elements were callables. Use a dict of score name mapped to the scorer callable. Got {scoring!r}')
                else:
                    raise ValueError(f'{err_msg} Non-string types were found in the given list. Got {scoring!r}')
            scorers = {scorer: check_scoring(estimator, scoring=scorer) for scorer in scoring}
        else:
            raise ValueError(f'{err_msg} Empty list was given. {scoring!r}')
    elif isinstance(scoring, dict):
        keys = set(scoring)
        if not all((isinstance(k, str) for k in keys)):
            raise ValueError(f'Non-string types were found in the keys of the given dict. scoring={scoring!r}')
        if len(keys) == 0:
            raise ValueError(f'An empty dict was passed. {scoring!r}')
        scorers = {key: check_scoring(estimator, scoring=scorer) for key, scorer in scoring.items()}
    else:
        raise ValueError(err_msg_generic)
    return scorers