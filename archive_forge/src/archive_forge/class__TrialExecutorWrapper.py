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
class _TrialExecutorWrapper:
    """Wraps around TrialExecutor class, intercepts API calls and warns users
    of restricted API access.

    This is meant to facilitate restricting
    the current API exposure of TrialExecutor by TrialScheduler.
    """

    def __init__(self, trial_executor: '_FakeRayTrialExecutor', whitelist_attr: Optional[set]=None):
        self._trial_executor = trial_executor
        self._whitelist_attr = whitelist_attr or set()
        for attr in self._whitelist_attr:
            assert hasattr(self._trial_executor, attr)

    def __getattr__(self, attr):
        if attr not in self._whitelist_attr:
            if log_once('restrict_accessing_trial_executor'):
                logger.warning(f'You are trying to access {attr} interface of TrialExecutor in TrialScheduler, which is being restricted. If you believe it is reasonable for your scheduler to access this TrialExecutor API, please reach out to Ray team on GitHub. A more strict API access pattern would be enforced starting 1.12.0')
        return getattr(self._trial_executor, attr)