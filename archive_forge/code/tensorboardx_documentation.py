import logging
import numpy as np
from typing import TYPE_CHECKING, Dict
from ray.air.constants import TRAINING_ITERATION
from ray.tune.logger.logger import _LOGGER_DEPRECATION_WARNING, Logger, LoggerCallback
from ray.util.debug import log_once
from ray.tune.result import (
from ray.tune.utils import flatten_dict
from ray.util.annotations import Deprecated, PublicAPI
TensorBoardX Logger.

    Note that hparams will be written only after a trial has terminated.
    This logger automatically flattens nested dicts to show on TensorBoard:

        {"a": {"b": 1, "c": 2}} -> {"a/b": 1, "a/c": 2}
    