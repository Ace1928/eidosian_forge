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
def _get_multiplier(self, increase_by: Dict[str, float], cpus: float=0, gpus: float=0, max_multiplier: int=-1) -> int:
    """Get how many times ``increase_by`` bundles
        occur in ``cpus`` and ``gpus``."""
    if increase_by.get('CPU', 0) and increase_by.get('GPU', 0):
        multiplier = min(cpus // increase_by.get('CPU', 0), gpus // increase_by.get('GPU', 0))
    elif increase_by.get('GPU', 0):
        multiplier = gpus // increase_by.get('GPU', 0)
    else:
        multiplier = cpus // increase_by.get('CPU', 0)
    if max_multiplier > 0 and multiplier > 0:
        multiplier = min(max_multiplier, multiplier)
    return int(multiplier)