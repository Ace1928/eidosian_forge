import functools
import logging
import os
import numpy as np
from mlflow.metrics.base import MetricValue, standard_aggregations
def _mae_eval_fn(predictions, targets=None, metrics=None, sample_weight=None):
    if targets is not None and len(targets) != 0:
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(targets, predictions, sample_weight=sample_weight)
        return MetricValue(aggregate_results={'mean_absolute_error': mae})