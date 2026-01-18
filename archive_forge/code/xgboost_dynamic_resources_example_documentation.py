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
This is a basic example of a resource allocating function.

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
        