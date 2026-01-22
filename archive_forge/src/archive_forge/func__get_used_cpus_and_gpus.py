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
def _get_used_cpus_and_gpus(self, t: Trial) -> Tuple[float, float]:
    """Check how many CPUs and GPUs a trial is using currently"""
    return (t.placement_group_factory.required_resources.get('CPU', 0), t.placement_group_factory.required_resources.get('GPU', 0))