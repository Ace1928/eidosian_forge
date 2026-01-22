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
class DistributeResources:
    """This class creates a basic uniform resource allocation function.

    The function naively balances free resources (CPUs and GPUs) between
    trials, giving them all equal priority, ensuring that all resources
    are always being used. The free resources will be placed in new bundles.
    The function assumes that all bundles are equal (there is no "head"
    bundle).

    If for some reason a trial ends up with
    more resources than there are free ones, it will adjust downwards.
    It will also ensure that trial as at least as many resources as
    it started with (``base_trial_resource``).

    The function returns a new ``PlacementGroupFactory`` with updated
    resource requirements, or None. If the returned
    ``PlacementGroupFactory`` is equal by value to the one the
    trial has currently, the scheduler will skip the update process
    internally (same with None).

    If you wish to implement your own resource distribution logic,
    you can do so by extending this class, as it provides several
    generic methods. You can also implement a function instead.

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
    """

    def __init__(self, add_bundles: bool=False, increase_by: Optional[Dict[str, float]]=None, increase_by_times: int=-1, reserve_resources: Optional[Dict[str, float]]=None):
        self.add_bundles = add_bundles
        self.increase_by = increase_by or {}
        self.increase_by_times = increase_by_times
        self.reserve_resources = reserve_resources or {}

    def _validate(self, base_trial_resource: PlacementGroupFactory, result: Dict[str, Any]) -> bool:
        """Return False if we should keep the current resources outright."""
        if not isinstance(base_trial_resource, PlacementGroupFactory):
            raise ValueError(f'{self.__class__.__name__} only supports PlacementGroupFactories.')
        if not self.add_bundles and len(base_trial_resource.bundles) > 1:
            raise ValueError(f'If `add_bundles` is False, the number of bundles in `resources_per_trial` must be 1 (got {len(base_trial_resource.bundles)}).')
        if result['training_iteration'] < 1:
            return False
        return True

    def _get_total_available_resources(self, tune_controller: 'TuneController') -> Tuple[float, float]:
        """Get the number of CPUs and GPUs avaialble in total (not just free)"""
        total_available_cpus = tune_controller._resource_updater.get_num_cpus() - self.reserve_resources.get('CPU', 0)
        total_available_gpus = tune_controller._resource_updater.get_num_gpus() - self.reserve_resources.get('GPU', 0)
        return (total_available_cpus, total_available_gpus)

    def _get_used_cpus_and_gpus(self, t: Trial) -> Tuple[float, float]:
        """Check how many CPUs and GPUs a trial is using currently"""
        return (t.placement_group_factory.required_resources.get('CPU', 0), t.placement_group_factory.required_resources.get('GPU', 0))

    def _get_resources_from_bundles(self, bundles: List[Dict[str, float]]) -> Dict[str, float]:
        """Get total sums of resources in bundles"""
        if not bundles:
            return {'CPU': 0, 'GPU': 0}
        return _sum_bundles(bundles)

    def _is_bundle_empty(self, bundle: Dict[str, float]) -> bool:
        return not (bundle.get('CPU', 0) or bundle.get('GPU', 0))

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

    def _create_new_bundles(self, increase_by: Dict[str, float], multiplier: int) -> List[Dict[str, float]]:
        """Create a list of new bundles containing ``increase_by`` * ``multiplier``."""
        multiplier = abs(multiplier)
        if self.add_bundles:
            bundles = [increase_by] * int(multiplier)
        else:
            bundles = [{}]
            bundles[0]['CPU'] = increase_by.get('CPU', 0) * multiplier
            bundles[0]['GPU'] = increase_by.get('GPU', 0) * multiplier
        return bundles

    def _modify_bundles_with_free_resources(self, bundles: List[Dict[str, float]], increase_by: Dict[str, float], free_cpus: float, free_gpus: float, *, max_multiplier: int=-1, max_increase_by_times: int=-1):
        """Given free resources, increase/decrease the number of bundles in
        ``bundles``."""
        multiplier = self._get_multiplier(increase_by, free_cpus, free_gpus, max_multiplier)
        if multiplier < 0:
            bundles = self._remove_bundles(bundles, increase_by, multiplier)
        elif multiplier > 0:
            bundles_to_add = self._create_new_bundles(increase_by, multiplier)
            bundles = self._add_two_bundles(bundles, bundles_to_add, increase_by, True, max_increase_by_times)
        return bundles

    def _get_added_bundles(self, bundles: List[Dict[str, float]], base_bundles: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Return the difference between bundles and base_bundles"""
        if self.add_bundles:
            added_bundles = bundles[len(base_bundles):]
        else:
            if not bundles:
                bundles = [{'CPU': 0, 'GPU': 0}]
            if not base_bundles:
                base_bundles = [{'CPU': 0, 'GPU': 0}]
            added_bundles = [{'CPU': bundles[0].get('CPU', 0) - base_bundles[0].get('CPU', 0), 'GPU': bundles[0].get('GPU', 0) - base_bundles[0].get('GPU', 0)}]
        return added_bundles

    def _are_bundles_below_limit(self, bundles: List[Dict[str, float]], base_bundles: Optional[List[Dict[str, float]]]=None, max_added_cpus: Optional[float]=None, max_added_gpus: Optional[float]=None):
        if not max_added_cpus:
            if self.increase_by_times > 0:
                max_added_cpus = self.increase_by.get('CPU', 0) * self.increase_by_times
            else:
                max_added_cpus = np.inf
        if not max_added_gpus:
            if self.increase_by_times > 0:
                max_added_gpus = self.increase_by.get('GPU', 0) * self.increase_by_times
            else:
                max_added_gpus = np.inf
        added_resources = self._get_resources_from_bundles(self._get_added_bundles(bundles, base_bundles) if base_bundles else bundles)
        ret = added_resources.get('CPU', -np.inf) < max_added_cpus or added_resources.get('GPU', -np.inf) < max_added_gpus
        return ret

    def _get_new_added_bundles(self, trial: Trial, all_trials: List[Trial], base_bundles: List[Dict[str, float]], increase_by: Dict[str, float], total_available_cpus: float, total_available_gpus: float, used_cpus: float, used_gpus: float) -> List[Dict[str, float]]:
        """Returns updated added bundles."""
        upper_limit_all_trials_bundles = [list() for _ in range(len(all_trials))]
        free_cpus = total_available_cpus - used_cpus
        free_gpus = total_available_gpus - used_gpus
        base_resources = self._get_resources_from_bundles(base_bundles)
        upper_limit_cpus_to_distribute = total_available_cpus - base_resources.get('CPU', 0) * len(all_trials)
        upper_limit_gpus_to_distribute = total_available_gpus - base_resources.get('GPU', 0) * len(all_trials)
        max_increase_by_times = 0
        i = 0
        trials_at_limit = set()
        while len(trials_at_limit) < len(all_trials) and upper_limit_cpus_to_distribute >= increase_by.get('CPU', 0) and (upper_limit_gpus_to_distribute >= increase_by.get('GPU', 0)):
            idx = i % len(upper_limit_all_trials_bundles)
            old_bundles = deepcopy(upper_limit_all_trials_bundles[idx])
            upper_limit_all_trials_bundles[idx] = self._modify_bundles_with_free_resources(upper_limit_all_trials_bundles[idx], increase_by, upper_limit_cpus_to_distribute, upper_limit_gpus_to_distribute, max_multiplier=1)
            added_resources = self._get_resources_from_bundles(self._get_added_bundles(upper_limit_all_trials_bundles[idx], old_bundles))
            if not added_resources.get('CPU', 0) and (not added_resources.get('GPU', 0)):
                trials_at_limit.add(idx)
            elif idx == 0:
                max_increase_by_times += 1
            upper_limit_cpus_to_distribute -= added_resources.get('CPU', 0)
            upper_limit_gpus_to_distribute -= added_resources.get('GPU', 0)
            i += 1
        return self._modify_bundles_with_free_resources(self._get_added_bundles(trial.placement_group_factory.bundles, base_bundles), increase_by, free_cpus, free_gpus, max_increase_by_times=max_increase_by_times)

    def __call__(self, tune_controller: 'TuneController', trial: Trial, result: Dict[str, Any], scheduler: 'ResourceChangingScheduler') -> Optional[PlacementGroupFactory]:
        """Run resource allocation logic.

        Returns a new ``PlacementGroupFactory`` with updated
        resource requirements, or None. If the returned
        ``PlacementGroupFactory`` is equal by value to the one the
        trial has currently, the scheduler will skip the update process
        internally (same with None).

        Args:
            tune_controller: Trial runner for this Tune run.
                Can be used to obtain information about other trials.
            trial: The trial to allocate new resources to.
            result: The latest results of trial.
            scheduler: The scheduler calling
                the function.
        """
        base_trial_resource = scheduler.base_trial_resources
        if not self._validate(base_trial_resource=base_trial_resource, result=result):
            return None
        if base_trial_resource is None:
            base_trial_resource = PlacementGroupFactory([{'CPU': 1, 'GPU': 0}])
        if self.increase_by:
            increase_by = self.increase_by
            assert not self._is_bundle_empty(increase_by)
            assert increase_by.get('CPU', 0) >= 0 and increase_by.get('GPU', 0) >= 0
        elif self.add_bundles:
            increase_by = base_trial_resource.bundles[-1]
        elif base_trial_resource.bundles[0].get('GPU', 0):
            increase_by = {'GPU': 1}
        else:
            increase_by = {'CPU': 1}
        base_bundles = deepcopy(base_trial_resource.bundles)
        total_available_cpus, total_available_gpus = self._get_total_available_resources(tune_controller=tune_controller)
        all_trials = tune_controller.get_live_trials()
        used_cpus_and_gpus = [self._get_used_cpus_and_gpus(t) for t in all_trials]
        used_cpus, used_gpus = zip(*used_cpus_and_gpus)
        used_cpus = sum(used_cpus)
        used_gpus = sum(used_gpus)
        added_bundles = self._get_new_added_bundles(trial, all_trials, base_bundles, increase_by, total_available_cpus, total_available_gpus, used_cpus, used_gpus)
        new_bundles = self._add_two_bundles(base_bundles, added_bundles, increase_by, False)
        pgf = PlacementGroupFactory(new_bundles, *base_trial_resource._args, strategy=base_trial_resource.strategy, **base_trial_resource._kwargs)
        pgf._head_bundle_is_empty = base_trial_resource._head_bundle_is_empty
        return pgf