import functools
import logging
import os
import numpy as np
from mlflow.metrics.base import MetricValue, standard_aggregations
def _ndcg_at_k_eval_fn(k):
    if not (isinstance(k, int) and k > 0):
        _logger.warning(f"Cannot calculate 'ndcg_at_k' for invalid parameter 'k'. 'k' should be a positive integer; found: {k}. Skipping metric logging.")
        return noop

    def _fn(predictions, targets):
        from sklearn.metrics import ndcg_score
        if not _validate_array_like_id_data(predictions, 'ndcg_at_k', predictions_col_specifier) or not _validate_array_like_id_data(targets, 'ndcg_at_k', targets_col_specifier):
            return
        scores = []
        for ground_truth, retrieved in zip(targets, predictions):
            if len(retrieved) == 0 and len(ground_truth) == 0:
                scores.append(1)
                continue
            if len(retrieved) == 0 or len(ground_truth) == 0:
                scores.append(0)
                continue
            y_score, y_true = _prepare_row_for_ndcg(retrieved[:k], ground_truth)
            score = ndcg_score(y_true, y_score, k=len(retrieved[:k]), ignore_ties=True)
            scores.append(score)
        return MetricValue(scores=scores, aggregate_results=standard_aggregations(scores))
    return _fn