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
@PublicAPI(stability='beta')
class DistributeResourcesToTopJob(DistributeResources):
    """This class creates a "TopJob" resource allocation function.

    The function will assign all of the free resources to the best
    performing trial (as defined by ``metric`` and ``mode``). The
    previous best trials will not have their resources deallocated,
    unless in the case outlined below.

    If for some reason a trial ends up with
    more resources than there are free ones, it will adjust downwards.
    It will also ensure that trial as at least as many resources as
    it started with (``base_trial_resource``).

    The function returns a new ``PlacementGroupFactory`` with updated
    resource requirements, or None. If the returned
    ``PlacementGroupFactory`` is equal by value to the one the
    trial has currently, the scheduler will skip the update process
    internally (same with None).

    Args:
        add_bundles: If True, create new bundles from free resources.
            Otherwise, spread them among base_trial_resource bundles.
        increase_by: A dict with key-value
            pairs representing an atomic unit of resources (name-amount)
            the trial will be increased by. If not set, the trial will
            increase by 1 CPU/GPU.
        increase_by_times: If set to >=1 and ``increase_by`` is set,
            the trial will increase by maximum of
            ``increase_by_times * increase_by`` resources. If set to <1,
            no upper limit is set. Ignored if ``increase_by`` is not set.
        reserve_resources: A dict of
            resource_name-amount pairs representing the resources
            that will not be allocated to resized trials.
            is that the attribute should increase monotonically.
        metric: The training result objective value attribute. Stopping
            procedures will use this attribute. If None, will use the metric
            of the scheduler.
        mode: One of {min, max}. Determines whether objective is
            minimizing or maximizing the metric attribute. If None, will use the metric
            of the scheduler.

    """

    def __init__(self, add_bundles: bool=False, increase_by: Optional[Dict[str, float]]=None, increase_by_times: int=-1, reserve_resources: Optional[Dict[str, float]]=None, metric: Optional[str]=None, mode: Optional[str]=None):
        super().__init__(add_bundles, increase_by, increase_by_times, reserve_resources)
        self.metric = metric
        self.mode = mode

    @property
    def _metric_op(self) -> float:
        if self.mode not in ('min', 'max'):
            raise ValueError('The mode parameter can only be either min or max.')
        if self.mode == 'max':
            return 1.0
        return -1.0

    def _get_new_added_bundles(self, trial: Trial, all_trials: List[Trial], base_bundles: List[Dict[str, float]], increase_by: Dict[str, float], total_available_cpus: float, total_available_gpus: float, used_cpus: float, used_gpus: float) -> List[Dict[str, float]]:
        if self.metric is None:
            raise ValueError('The metric parameter cannot be None. The parameter can be set in either `DistributeResourcesToTopJob`, the base scheduler or in `tune.TuneConfig()` (highest to lowest priority).')
        free_cpus = total_available_cpus - used_cpus
        free_gpus = total_available_gpus - used_gpus
        sorted_trials = sorted(all_trials, key=lambda t: -self._metric_op * t.last_result.get(self.metric, np.inf))
        added_bundles = self._get_added_bundles(trial.placement_group_factory.bundles, base_bundles)
        best_trial = next((t for t in sorted_trials if self._are_bundles_below_limit(t.placement_group_factory.bundles, base_bundles)), sorted_trials[0])
        if trial.trial_id != best_trial.trial_id and self._get_multiplier(increase_by, free_cpus, free_gpus) >= 0:
            return added_bundles
        return self._modify_bundles_with_free_resources(added_bundles, increase_by, free_cpus, free_gpus)