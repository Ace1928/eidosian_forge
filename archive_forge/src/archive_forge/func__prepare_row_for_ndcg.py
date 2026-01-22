import functools
import logging
import os
import numpy as np
from mlflow.metrics.base import MetricValue, standard_aggregations
def _prepare_row_for_ndcg(predictions, targets):
    """Prepare data one row from predictions and targets to y_score, y_true for ndcg calculation.

    Args:
        predictions: A list of strings of at most k doc IDs retrieved.
        targets: A list of strings of ground-truth doc IDs.

    Returns:
        y_true : ndarray of shape (1, n_docs) Representing the ground-truth relevant docs.
            n_docs is the number of unique docs in union of predictions and targets.
        y_score : ndarray of shape (1, n_docs) Representing the retrieved docs.
            n_docs is the number of unique docs in union of predictions and targets.
    """
    eps = 1e-06
    targets = set(targets)
    predictions, targets = _expand_duplicate_retrieved_docs(predictions, targets)
    all_docs = targets.union(predictions)
    doc_id_to_index = {doc_id: i for i, doc_id in enumerate(all_docs)}
    n_labels = max(len(doc_id_to_index), 2)
    y_true = np.zeros((1, n_labels), dtype=np.float32)
    y_score = np.zeros((1, n_labels), dtype=np.float32)
    for i, doc_id in enumerate(predictions):
        y_score[0, doc_id_to_index[doc_id]] = 1 - i * eps
    for doc_id in targets:
        y_true[0, doc_id_to_index[doc_id]] = 1
    return (y_score, y_true)