import copy
import glob
import logging
import os
import warnings
from typing import Dict, Optional, List, Union, Any, TYPE_CHECKING
from ray.air._internal.usage import tag_searcher
from ray.tune.search.util import _set_search_properties_backwards_compatible
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.debug import log_once
def _set_searcher_max_concurrency(self):
    self._limit_concurrency = not self.searcher.set_max_concurrency(self.max_concurrent)