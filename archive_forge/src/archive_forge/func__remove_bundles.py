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
def _remove_bundles(self, bundles: List[Dict[str, float]], increase_by: Dict[str, float], multiplier: int) -> List[Dict[str, float]]:
    """Remove ``multiplier`` ``increase_by`` bundles from ``bundles``."""
    multiplier = -abs(multiplier)
    if self.add_bundles:
        bundles = bundles[:multiplier]
    else:
        bundles = deepcopy(bundles)
        bundles[0]['CPU'] += increase_by.get('CPU', 0) * multiplier
        bundles[0]['GPU'] += increase_by.get('GPU', 0) * multiplier
        bundles[0]['CPU'] = max(bundles[0]['CPU'], 0)
        bundles[0]['GPU'] = max(bundles[0]['GPU'], 0)
    return bundles