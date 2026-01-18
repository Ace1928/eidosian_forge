import json
import logging
import numpy as np
import os
from typing import TYPE_CHECKING, Dict, TextIO
from ray.air.constants import (
import ray.cloudpickle as cloudpickle
from ray.tune.logger.logger import _LOGGER_DEPRECATION_WARNING, Logger, LoggerCallback
from ray.tune.utils.util import SafeFallbackEncoder
from ray.util.annotations import Deprecated, PublicAPI
Logs trial results in json format.

    Also writes to a results file and param.json file when results or
    configurations are updated. Experiments must be executed with the
    JsonLoggerCallback to be compatible with the ExperimentAnalysis tool.
    