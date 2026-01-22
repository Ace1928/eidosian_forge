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
def _validate_result_metrics(self, result):
    """
        Check if any of the required metrics was not reported
        in the last result. If the only items are ``done`` or any of
        DEBUG_METRICS, this means that no result was ever received and
        the trial just returned. This is also okay and will not raise
        an error.

        This will ignore checking for the DEFAULT_METRIC.
        """
    if int(os.environ.get('TUNE_DISABLE_STRICT_METRIC_CHECKING', 0)) != 1 and len({k for k in result if k not in list(DEBUG_METRICS) + [DONE]}) > 1:
        base_metric = self._metric if self._metric != DEFAULT_METRIC else None
        scheduler_metric = self._scheduler_alg.metric if self._scheduler_alg.metric != DEFAULT_METRIC else None
        search_metrics = self._search_alg.metric if self._search_alg.metric != DEFAULT_METRIC else None
        if isinstance(search_metrics, str):
            search_metrics = [search_metrics]
        if base_metric and base_metric not in result:
            report_metric = base_metric
            location = 'tune.TuneConfig()'
        elif scheduler_metric and scheduler_metric not in result:
            report_metric = scheduler_metric
            location = type(self._scheduler_alg).__name__
        elif search_metrics and any((search_metric not in result for search_metric in search_metrics)):
            report_metric = list(filter(lambda search_metric: search_metric not in result, search_metrics))
            if len(report_metric) == 1:
                report_metric = report_metric[0]
            location = type(self._search_alg).__name__
        else:
            report_metric = None
            location = None
        if report_metric:
            raise ValueError('Trial returned a result which did not include the specified metric(s) `{}` that `{}` expects. Make sure your calls to `tune.report()` include the metric, or set the TUNE_DISABLE_STRICT_METRIC_CHECKING environment variable to 1. Result: {}'.format(report_metric, location, result))