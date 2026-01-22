import logging
import math
import time
import warnings
from collections import defaultdict
from typing import Dict, List
import numpy as np
import pandas
import ray
import xgboost as xgb
from ray.util import get_node_ip_address
from modin.core.execution.ray.common import RayWrapper
from modin.distributed.dataframe.pandas import from_partitions
from .utils import RabitContext, RabitContextManager
def _get_cpus_per_actor(num_actors):
    """
    Get number of CPUs to use by each actor.

    Parameters
    ----------
    num_actors : int
        Number of Ray actors.

    Returns
    -------
    int
        Number of CPUs per actor.
    """
    cluster_cpus = _get_cluster_cpus()
    cpus_per_actor = max(1, min(int(_get_min_cpus_per_node() or 1), int(cluster_cpus // num_actors)))
    return cpus_per_actor