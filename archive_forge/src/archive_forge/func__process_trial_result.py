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
def _process_trial_result(self, trial, result):
    result.update(trial_id=trial.trial_id)
    is_duplicate = RESULT_DUPLICATE in result
    force_checkpoint = result.get(SHOULD_CHECKPOINT, False)
    if is_duplicate:
        logger.debug("Trial finished without logging 'done'.")
        result = trial.last_result
        result.update(done=True)
    self._total_time += result.get(TIME_THIS_ITER_S, 0)
    flat_result = flatten_dict(result)
    self._validate_result_metrics(flat_result)
    if self._stopper(trial.trial_id, result) or trial.should_stop(flat_result):
        decision = TrialScheduler.STOP
    else:
        with warn_if_slow('scheduler.on_trial_result'):
            decision = self._scheduler_alg.on_trial_result(self._wrapped(), trial, flat_result)
    if decision == TrialScheduler.STOP:
        result.update(done=True)
    else:
        with warn_if_slow('search_alg.on_trial_result'):
            self._search_alg.on_trial_result(trial.trial_id, flat_result)
    if not is_duplicate:
        with warn_if_slow('callbacks.on_trial_result'):
            self._callbacks.on_trial_result(iteration=self._iteration, trials=self._trials, trial=trial, result=result.copy())
        trial.update_last_result(result)
        self._mark_trial_to_checkpoint(trial)
    if decision != TrialScheduler.PAUSE:
        self._checkpoint_trial_if_needed(trial, force=force_checkpoint)
    if trial.is_saving:
        logger.debug(f'Caching trial decision for trial {trial}: {decision}')
        if not self._cached_trial_decisions.get(trial.trial_id) or decision in {TrialScheduler.PAUSE, TrialScheduler.STOP}:
            self._cached_trial_decisions[trial.trial_id] = decision
        return None
    else:
        self._queue_decision(trial, decision)
        return decision