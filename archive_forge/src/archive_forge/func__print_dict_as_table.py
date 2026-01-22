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
def _print_dict_as_table(data: Dict, header: Optional[str]=None, include: Optional[Collection[str]]=None, exclude: Optional[Collection[str]]=None, division: Optional[Collection[str]]=None):
    table_data = _get_dict_as_table_data(data=data, include=include, exclude=exclude, upper_keys=division)
    headers = [header, ''] if header else []
    if not table_data:
        return
    print(tabulate(table_data, headers=headers, colalign=('left', 'right'), tablefmt=AIR_TABULATE_TABLEFMT))