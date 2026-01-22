import logging
import os
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from typing import Any, Dict, List, Optional, Union
from mlflow.entities import Metric, Param, RunTag
from mlflow.entities.dataset_input import DatasetInput
from mlflow.exceptions import MlflowException
from mlflow.tracking.client import MlflowClient
from mlflow.utils import _truncate_dict, chunk_list
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import (
def _try_operation(self, fn, *args, **kwargs):
    """
        Attempt to evaluate the specified function, `fn`, on the specified `*args` and `**kwargs`,
        returning either the result of the function evaluation (if evaluation was successful) or
        the exception raised by the function evaluation (if evaluation was unsuccessful).
        """
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return e