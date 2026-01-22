import functools
import logging
import os
import numpy as np
from mlflow.metrics.base import MetricValue, standard_aggregations
def _f1_score_eval_fn(predictions, targets=None, metrics=None, pos_label=1, average='binary', sample_weight=None):
    if targets is not None and len(targets) != 0:
        from sklearn.metrics import f1_score
        f1 = f1_score(targets, predictions, pos_label=pos_label, average=average, sample_weight=sample_weight)
        return MetricValue(aggregate_results={'f1_score': f1})