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
class _MultimetricScorer:
    """Callable for multimetric scoring used to avoid repeated calls
    to `predict_proba`, `predict`, and `decision_function`.

    `_MultimetricScorer` will return a dictionary of scores corresponding to
    the scorers in the dictionary. Note that `_MultimetricScorer` can be
    created with a dictionary with one key  (i.e. only one actual scorer).

    Parameters
    ----------
    scorers : dict
        Dictionary mapping names to callable scorers.

    raise_exc : bool, default=True
        Whether to raise the exception in `__call__` or not. If set to `False`
        a formatted string of the exception details is passed as result of
        the failing scorer.
    """

    def __init__(self, *, scorers, raise_exc=True):
        self._scorers = scorers
        self._raise_exc = raise_exc

    def __call__(self, estimator, *args, **kwargs):
        """Evaluate predicted target values."""
        scores = {}
        cache = {} if self._use_cache(estimator) else None
        cached_call = partial(_cached_call, cache)
        if _routing_enabled():
            routed_params = process_routing(self, 'score', **kwargs)
        else:
            routed_params = Bunch(**{name: Bunch(score=kwargs) for name in self._scorers})
        for name, scorer in self._scorers.items():
            try:
                if isinstance(scorer, _BaseScorer):
                    score = scorer._score(cached_call, estimator, *args, **routed_params.get(name).score)
                else:
                    score = scorer(estimator, *args, **routed_params.get(name).score)
                scores[name] = score
            except Exception as e:
                if self._raise_exc:
                    raise e
                else:
                    scores[name] = format_exc()
        return scores

    def _use_cache(self, estimator):
        """Return True if using a cache is beneficial, thus when a response method will
        be called several time.
        """
        if len(self._scorers) == 1:
            return False
        counter = Counter([_check_response_method(estimator, scorer._response_method).__name__ for scorer in self._scorers.values() if isinstance(scorer, _BaseScorer)])
        if any((val > 1 for val in counter.values())):
            return True
        return False

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.3

        Returns
        -------
        routing : MetadataRouter
            A :class:`~utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        return MetadataRouter(owner=self.__class__.__name__).add(**self._scorers, method_mapping='score')