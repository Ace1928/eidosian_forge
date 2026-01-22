from __future__ import print_function
import collections
import datetime
import numbers
import os
import sys
import textwrap
import time
import warnings
from typing import Any, Callable, Collection, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import ray
from ray._private.dict import flatten_dict
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.experimental.tqdm_ray import safe_print
from ray.air.util.node import _force_on_current_node
from ray.air.constants import EXPR_ERROR_FILE, TRAINING_ITERATION
from ray.tune.callback import Callback
from ray.tune.logger import pretty_print
from ray.tune.result import (
from ray.tune.experiment.trial import DEBUG_PRINT_INTERVAL, Trial, _Location
from ray.tune.trainable import Trainable
from ray.tune.utils import unflattened_lookup
from ray.tune.utils.log import Verbosity, has_verbosity, set_verbosity
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.queue import Empty, Queue
from ray.widgets import Template
def _time_passed_str(start_time: float, current_time: float) -> str:
    """Generate a message describing the current and elapsed time in the run.

    Args:
        start_time: POSIX timestamp of the start of the tune run
        current_time: POSIX timestamp giving the current time

    Returns:
        Message with the current and elapsed time for the current tune run,
            formatted to be displayed to the user
    """
    current_time_str, running_for_str = _get_time_str(start_time, current_time)
    return f'Current time: {current_time_str} (running for {running_for_str})'