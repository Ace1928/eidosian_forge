import copy
import logging
from typing import Dict, Optional, List
from ray.tune.search.searcher import Searcher
from ray.tune.search.util import _set_search_properties_backwards_compatible
from ray.util.annotations import PublicAPI
A wrapper algorithm for limiting the number of concurrent trials.

    Certain Searchers have their own internal logic for limiting
    the number of concurrent trials. If such a Searcher is passed to a
    ``ConcurrencyLimiter``, the ``max_concurrent`` of the
    ``ConcurrencyLimiter`` will override the ``max_concurrent`` value
    of the Searcher. The ``ConcurrencyLimiter`` will then let the
    Searcher's internal logic take over.

    Args:
        searcher: Searcher object that the
            ConcurrencyLimiter will manage.
        max_concurrent: Maximum concurrent samples from the underlying
            searcher.
        batch: Whether to wait for all concurrent samples
            to finish before updating the underlying searcher.

    Example:

    .. code-block:: python

        from ray.tune.search import ConcurrencyLimiter
        search_alg = HyperOptSearch(metric="accuracy")
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2)
        tuner = tune.Tuner(
            trainable,
            tune_config=tune.TuneConfig(
                search_alg=search_alg
            ),
        )
        tuner.fit()
    