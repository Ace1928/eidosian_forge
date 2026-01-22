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
def _get_resources_from_bundles(self, bundles: List[Dict[str, float]]) -> Dict[str, float]:
    """Get total sums of resources in bundles"""
    if not bundles:
        return {'CPU': 0, 'GPU': 0}
    return _sum_bundles(bundles)