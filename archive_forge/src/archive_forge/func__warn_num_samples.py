import copy
import logging
from typing import Dict, List, Optional
import numpy as np
from ray.tune.search import Searcher
from ray.tune.search.util import _set_search_properties_backwards_compatible
from ray.util import PublicAPI
def _warn_num_samples(searcher: Searcher, num_samples: int):
    if isinstance(searcher, Repeater) and num_samples % searcher.repeat:
        logger.warning('`num_samples` is now expected to be the total number of trials, including the repeat trials. For example, set num_samples=15 if you intend to obtain 3 search algorithm suggestions and repeat each suggestion 5 times. Any leftover trials (num_samples mod repeat) will be ignored.')