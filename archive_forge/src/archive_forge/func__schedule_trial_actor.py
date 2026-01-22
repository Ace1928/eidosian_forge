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
def _schedule_trial_actor(self, trial: Trial):
    """Schedule an actor for a trial.

        If a cached actor is available, use it. Otherwise, request a
        new actor.
        """
    logger.debug(f'Trying to schedule new ACTOR for trial {trial}')
    assert trial.status == Trial.PENDING
    trial.init_local_path()
    self._mark_trial_to_checkpoint(trial)
    if self._maybe_reuse_cached_actor(trial):
        return
    if trial in self._trial_to_actor:
        raise RuntimeError(f'Tried to request a new actor for trial {trial}, but an old actor still exists. This can lead to leaked resources. The old actor should be removed first. This is an internal problem in Ray Tune. If you encounter this error, please raise an issue on https://github.com/ray-project/ray/issues')
    trainable_cls = trial.get_trainable_cls()
    if not trainable_cls:
        exception = _AbortTrialExecution(f'Invalid trainable: {trial.trainable_name}. If you passed a string, make sure the trainable was registered before.')
        trial.handle_error(exception)
        self._schedule_trial_stop(trial, exception=exception)
        return
    _actor_cls = self._class_cache.get(trainable_cls)
    trial.set_location(_Location())
    trainable_kwargs = _get_trainable_kwargs(trial=trial)
    with _change_working_directory(trial):
        tracked_actor = self._actor_manager.add_actor(cls=_actor_cls, resource_request=trial.placement_group_factory, kwargs=trainable_kwargs, on_start=self._actor_started, on_stop=self._actor_stopped, on_error=self._actor_failed)
        self._trial_to_actor[trial] = tracked_actor
        self._actor_to_trial[tracked_actor] = trial
    logger.debug(f'Scheduled new ACTOR for trial {trial}: {tracked_actor}. Resources: {trial.placement_group_factory}')