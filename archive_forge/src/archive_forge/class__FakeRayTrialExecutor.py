import copy
import json
import time
import traceback
import uuid
import warnings
from collections import defaultdict, deque
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, Set
import logging
import os
import ray
from ray.air import ResourceRequest
from ray.air.constants import TIME_THIS_ITER_S
from ray.air.execution import ResourceManager, PlacementGroupResourceManager
from ray.air.execution._internal import RayActorManager, TrackedActor
from ray.train import CheckpointConfig
from ray.train._internal.session import _FutureTrainingResult
from ray.train._internal.storage import StorageContext
from ray.exceptions import RayActorError, RayTaskError
from ray.tune.error import _AbortTrialExecution, _TuneStopTrialError
from ray.tune.execution.class_cache import _ActorClassCache
from ray.tune.execution.experiment_state import (
from ray.tune.experiment.trial import (
from ray.tune.experiment import Experiment
from ray.tune.execution.insufficient_resources_manager import (
from ray.tune.result import (
from ray.tune.result import TRIAL_INFO, STDOUT_FILE, STDERR_FILE
from ray.tune import TuneError
from ray.tune.callback import Callback, CallbackList
from ray.tune.schedulers import FIFOScheduler, TrialScheduler
from ray.tune.stopper import NoopStopper, Stopper
from ray.tune.search import BasicVariantGenerator, SearchAlgorithm
from ray.tune.experiment import Trial
from ray.tune.utils.log import _dedup_logs
from ray.tune.utils.object_cache import _ObjectCache
from ray.tune.utils.resource_updater import _ResourceUpdater
from ray.tune.utils import warn_if_slow, flatten_dict
from ray.tune.utils.log import Verbosity, has_verbosity
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.tune.utils.serialization import TuneFunctionDecoder, TuneFunctionEncoder
from ray.util.annotations import DeveloperAPI, Deprecated
from ray.util.debug import log_once
class _FakeRayTrialExecutor:
    """The TuneController does not use a RayTrialExecutor anymore.

    Instead, we pass this fake executor for searchers/schedulers to use
    as an interface.

    In the future, we should have the searchers/schedulers either interact with
    the tune controller, or define a different API for more fine-grained scheduler
    control.
    """

    def __init__(self, tune_controller: TuneController):
        self._tune_controller = tune_controller

    def pause_trial(self, trial: Trial, should_checkpoint: bool=True):
        return self._tune_controller._schedule_trial_pause(trial, should_checkpoint=should_checkpoint)

    def save(self, trial: Trial, result: Optional[Dict]=None) -> Optional[_FutureTrainingResult]:
        return self._tune_controller._schedule_trial_save(trial=trial, result=result)

    def has_resources_for_trial(self, trial: Trial):
        return True

    @property
    def _resource_updater(self):
        return self._tune_controller._resource_updater

    def force_reconcilation_on_next_step_end(self):
        pass