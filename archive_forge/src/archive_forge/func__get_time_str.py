import argparse
import sys
from typing import (
import collections
from dataclasses import dataclass
import datetime
from enum import IntEnum
import logging
import math
import numbers
import numpy as np
import os
import pandas as pd
import textwrap
import time
from ray.air._internal.usage import AirEntrypoint
from ray.train import Checkpoint
from ray.tune.search.sample import Domain
from ray.tune.utils.log import Verbosity
import ray
from ray._private.dict import unflattened_lookup, flatten_dict
from ray._private.thirdparty.tabulate.tabulate import (
from ray.air.constants import TRAINING_ITERATION
from ray.tune.callback import Callback
from ray.tune.result import (
from ray.tune.experiment.trial import Trial
def _get_time_str(start_time: float, current_time: float) -> Tuple[str, str]:
    """Get strings representing the current and elapsed time.

    Args:
        start_time: POSIX timestamp of the start of the tune run
        current_time: POSIX timestamp giving the current time

    Returns:
        Current time and elapsed time for the current run
    """
    current_time_dt = datetime.datetime.fromtimestamp(current_time)
    start_time_dt = datetime.datetime.fromtimestamp(start_time)
    delta: datetime.timedelta = current_time_dt - start_time_dt
    rest = delta.total_seconds()
    days = int(rest // (60 * 60 * 24))
    rest -= days * (60 * 60 * 24)
    hours = int(rest // (60 * 60))
    rest -= hours * (60 * 60)
    minutes = int(rest // 60)
    seconds = int(rest - minutes * 60)
    running_for_str = ''
    if days > 0:
        running_for_str += f'{days:d}d '
    if hours > 0 or running_for_str:
        running_for_str += f'{hours:d}hr '
    if minutes > 0 or running_for_str:
        running_for_str += f'{minutes:d}min '
    running_for_str += f'{seconds:d}s'
    return (f'{current_time_dt:%Y-%m-%d %H:%M:%S}', running_for_str)