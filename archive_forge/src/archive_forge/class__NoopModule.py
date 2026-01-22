import logging
from types import ModuleType
from typing import Dict, Optional, Union
import ray
from ray.air import session
from ray.air._internal.mlflow import _MLflowLoggerUtil
from ray.air._internal import usage as air_usage
from ray.air.constants import TRAINING_ITERATION
from ray.tune.logger import LoggerCallback
from ray.tune.result import TIMESTEPS_TOTAL
from ray.tune.experiment import Trial
from ray.util.annotations import PublicAPI
class _NoopModule:

    def __getattr__(self, item):
        return _NoopModule()

    def __call__(self, *args, **kwargs):
        return None