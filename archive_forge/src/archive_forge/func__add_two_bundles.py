from copy import deepcopy
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Set, Tuple, Union, Callable, TYPE_CHECKING
import pickle
import warnings
from ray.air.execution.resources.request import _sum_bundles
from ray.util.annotations import PublicAPI
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.experiment import Trial
from ray.tune.execution.placement_groups import PlacementGroupFactory
def _add_two_bundles(self, bundles_a: List[Dict[str, float]], bundles_b: List[Dict[str, float]], increase_by: Dict[str, float], limit_to_increase_by_times: bool, max_increase_by_times: int=-1):
    """Add two bundles together.

        If ``limit_to_increase_by_times`` is True, ``self.increase_by_times`` > 0
        and ``max_increase_by_times`` > 0, ensure that the resulting number of
        bundles is not above ``min(max_increase_by_times, self.increase_by_times)``.

        If ``limit_to_increase_by_times`` is True and ``self.increase_by_times`` > 0,
        ensure that the resulting number of bundles is not above
        `self.increase_by_times``.
        """
    if limit_to_increase_by_times:
        if max_increase_by_times > 0 and self.increase_by_times > 0:
            max_increase_by_times = min(max_increase_by_times, self.increase_by_times)
        elif self.increase_by_times > 0:
            max_increase_by_times = self.increase_by_times
    if self.add_bundles:
        bundles = [b for b in bundles_a if not self._is_bundle_empty(b)] + [b for b in bundles_b if not self._is_bundle_empty(b)]
        if max_increase_by_times > 0:
            bundles = bundles[:max_increase_by_times]
    else:
        bundles_a = bundles_a or [{}]
        bundles_b = bundles_b or [{}]
        bundles = [{'CPU': bundles_a[0].get('CPU', 0) + bundles_b[0].get('CPU', 0), 'GPU': bundles_a[0].get('GPU', 0) + bundles_b[0].get('GPU', 0)}]
        if max_increase_by_times > 0:
            bundles[0]['CPU'] = min(bundles[0]['CPU'], increase_by.get('CPU', 0) * max_increase_by_times)
            bundles[0]['GPU'] = min(bundles[0]['GPU'], increase_by.get('GPU', 0) * max_increase_by_times)
    return bundles