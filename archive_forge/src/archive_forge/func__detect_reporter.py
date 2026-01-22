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
def _detect_reporter(verbosity: AirVerbosity, num_samples: int, entrypoint: Optional[AirEntrypoint]=None, metric: Optional[str]=None, mode: Optional[str]=None, config: Optional[Dict]=None, progress_metrics: Optional[Union[List[str], List[Dict[str, str]]]]=None):
    if entrypoint in {AirEntrypoint.TUNE_RUN, AirEntrypoint.TUNE_RUN_EXPERIMENTS, AirEntrypoint.TUNER}:
        reporter = TuneTerminalReporter(verbosity, num_samples=num_samples, metric=metric, mode=mode, config=config, progress_metrics=progress_metrics)
    else:
        reporter = TrainReporter(verbosity, progress_metrics=progress_metrics)
    return reporter