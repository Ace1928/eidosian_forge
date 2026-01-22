import functools
import logging
import os
import numpy as np
from mlflow.metrics.base import MetricValue, standard_aggregations
def _r2_score_eval_fn(predictions, targets=None, metrics=None, sample_weight=None):
    if targets is not None and len(targets) != 0:
        from sklearn.metrics import r2_score
        r2 = r2_score(targets, predictions, sample_weight=sample_weight)
        return MetricValue(aggregate_results={'r2_score': r2})