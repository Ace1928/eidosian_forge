from typing import Dict, Any, Optional, TYPE_CHECKING
import sklearn.datasets
import sklearn.metrics
import os
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost.core import Booster
import pickle
import ray
from ray import train, tune
from ray.tune.schedulers import ResourceChangingScheduler, ASHAScheduler
from ray.tune import Trainable
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.tune.experiment import Trial
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
def tune_xgboost(use_class_trainable=True):
    search_space = {'objective': 'binary:logistic', 'eval_metric': ['logloss', 'error'], 'max_depth': 9, 'learning_rate': 1, 'min_child_weight': tune.grid_search([2, 3]), 'subsample': tune.grid_search([0.8, 0.9]), 'colsample_bynode': tune.grid_search([0.8, 0.9]), 'random_state': 1, 'num_parallel_tree': 2000}
    base_scheduler = ASHAScheduler(max_t=16, grace_period=1, reduction_factor=2)

    def example_resources_allocation_function(tune_controller: 'TuneController', trial: Trial, result: Dict[str, Any], scheduler: 'ResourceChangingScheduler') -> Optional[PlacementGroupFactory]:
        """This is a basic example of a resource allocating function.

        The function naively balances available CPUs over live trials.

        This function returns a new ``PlacementGroupFactory`` with updated
        resource requirements, or None. If the returned
        ``PlacementGroupFactory`` is equal by value to the one the
        trial has currently, the scheduler will skip the update process
        internally (same with None).

        See :class:`DistributeResources` for a more complex,
        robust approach.

        Args:
            tune_controller: Trial runner for this Tune run.
                Can be used to obtain information about other trials.
            trial: The trial to allocate new resources to.
            result: The latest results of trial.
            scheduler: The scheduler calling the function.
        """
        base_trial_resource = scheduler._base_trial_resources
        if result['training_iteration'] < 1:
            return None
        if base_trial_resource is None:
            base_trial_resource = PlacementGroupFactory([{'CPU': 1, 'GPU': 0}])
        min_cpu = base_trial_resource.required_resources.get('CPU', 0)
        total_available_cpus = tune_controller._resource_updater.get_num_cpus()
        cpu_to_use = max(min_cpu, total_available_cpus // len(tune_controller.get_live_trials()))
        return PlacementGroupFactory([{'CPU': cpu_to_use, 'GPU': 0}])
    scheduler = ResourceChangingScheduler(base_scheduler=base_scheduler, resources_allocation_function=example_resources_allocation_function)
    if use_class_trainable:
        fn = BreastCancerTrainable
    else:
        fn = train_breast_cancer
    tuner = tune.Tuner(tune.with_resources(fn, resources=PlacementGroupFactory([{'CPU': 1, 'GPU': 0}])), tune_config=tune.TuneConfig(metric='eval-logloss', mode='min', num_samples=1, scheduler=scheduler), run_config=train.RunConfig(checkpoint_config=train.CheckpointConfig(checkpoint_at_end=use_class_trainable)), param_space=search_space)
    results = tuner.fit()
    if use_class_trainable:
        assert results.get_dataframe()['nthread'].max() > 1
    return results.get_best_result()