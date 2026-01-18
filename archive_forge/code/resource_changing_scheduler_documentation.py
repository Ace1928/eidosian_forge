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
Calls user defined resources_allocation_function. If the returned
        resources are not none and not the same as currently present, returns
        them. Otherwise, returns None.