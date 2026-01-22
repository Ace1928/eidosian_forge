import logging
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, List, Union
from ray.air.constants import TRAINING_ITERATION
from ray.tune.logger.logger import LoggerCallback
from ray.tune.result import (
from ray.tune.utils import flatten_dict
from ray.util.annotations import PublicAPI
Initializes an Aim Run object for a given trial.

        Args:
            trial: The Tune trial that aim will track as a Run.

        Returns:
            Run: The created aim run for a specific trial.
        