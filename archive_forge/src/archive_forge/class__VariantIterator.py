import copy
import glob
import itertools
import os
import uuid
from typing import Dict, List, Optional, Union, TYPE_CHECKING
import warnings
import numpy as np
from ray.air._internal.usage import tag_searcher
from ray.tune.error import TuneError
from ray.tune.experiment.config_parser import _make_parser, _create_trial_from_spec
from ray.tune.search.sample import np_random_generator, _BackwardsCompatibleNumpyRng
from ray.tune.search.variant_generator import (
from ray.tune.search.search_algorithm import SearchAlgorithm
from ray.tune.utils.util import _atomic_save, _load_newest_checkpoint
from ray.util import PublicAPI
class _VariantIterator:
    """Iterates over generated variants from the search space.

    This object also toggles between lazy evaluation and
    eager evaluation of samples. If lazy evaluation is enabled,
    this object cannot be serialized.
    """

    def __init__(self, iterable, lazy_eval=False):
        self.lazy_eval = lazy_eval
        self.iterable = iterable
        self._has_next = True
        if lazy_eval:
            self._load_value()
        else:
            self.iterable = list(iterable)
            self._has_next = bool(self.iterable)

    def _load_value(self):
        try:
            self.next_value = next(self.iterable)
        except StopIteration:
            self._has_next = False

    def has_next(self):
        return self._has_next

    def __next__(self):
        if self.lazy_eval:
            current_value = self.next_value
            self._load_value()
            return current_value
        current_value = self.iterable.pop(0)
        self._has_next = bool(self.iterable)
        return current_value