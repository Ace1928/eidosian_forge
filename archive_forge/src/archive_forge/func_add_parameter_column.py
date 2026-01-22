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
def add_parameter_column(self, parameter: str, representation: Optional[str]=None):
    """Adds a parameter to the existing columns.

        Args:
            parameter: Parameter to add. This must be a parameter
                specified in the configuration.
            representation: Representation to use in table. Defaults to
                `parameter`.
        """
    if parameter in self._parameter_columns:
        raise ValueError('Column {} already exists.'.format(parameter))
    if isinstance(self._parameter_columns, MutableMapping):
        representation = representation or parameter
        self._parameter_columns[parameter] = representation
    else:
        if representation is not None and representation != parameter:
            raise ValueError('`representation` cannot differ from `parameter` if this reporter was initialized with a list of metric columns.')
        self._parameter_columns.append(parameter)