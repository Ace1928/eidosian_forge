import functools
import logging
import os
import numpy as np
from mlflow.metrics.base import MetricValue, standard_aggregations
@functools.lru_cache(maxsize=8)
def _cached_evaluate_load(path, module_type=None):
    import evaluate
    return evaluate.load(path, module_type=module_type)