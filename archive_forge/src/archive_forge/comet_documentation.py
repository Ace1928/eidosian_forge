import os
from typing import Dict, List
import pyarrow.fs
from ray.tune.logger import LoggerCallback
from ray.tune.experiment import Trial
from ray.tune.utils import flatten_dict

        Log the current result of a Trial upon each iteration.
        