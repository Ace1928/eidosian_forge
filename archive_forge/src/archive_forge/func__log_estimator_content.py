import collections
import inspect
import logging
import pkgutil
import platform
import warnings
from copy import deepcopy
from importlib import import_module
from numbers import Number
from operator import itemgetter
import numpy as np
from packaging.version import Version
from mlflow import MlflowClient
from mlflow.utils.arguments_utils import _get_arg_names
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from mlflow.utils.time import get_current_time_millis
def _log_estimator_content(autologging_client, estimator, run_id, prefix, X, y_true=None, sample_weight=None, pos_label=None):
    """
    Logs content for the given estimator, which includes metrics and artifacts that might be
    tailored to the estimator's type (e.g., regression vs classification). Training labels
    are required for metric computation; metrics will be omitted if labels are not available.

    Args:
        autologging_client: An instance of `MlflowAutologgingQueueingClient` used for
            efficiently logging run data to MLflow Tracking.
        estimator: The estimator used to compute metrics and artifacts.
        run_id: The run under which the content is logged.
        prefix: A prefix used to name the logged content. Typically it's 'training_' for
            training-time content and user-controlled for evaluation-time content.
        X: The data samples.
        y_true: Labels.
        sample_weight: Per-sample weights used in the computation of metrics and artifacts.
        pos_label: The positive label used to compute binary classification metrics such as
            precision, recall, f1, etc. This parameter is only used for classification metrics.
            If set to `None`, the function will calculate metrics for each label and find their
            average weighted by support (number of true instances for each label).

    Returns:
        A dict of the computed metrics.
    """
    metrics = _log_specialized_estimator_content(autologging_client=autologging_client, fitted_estimator=estimator, run_id=run_id, prefix=prefix, X=X, y_true=y_true, sample_weight=sample_weight, pos_label=pos_label)
    if hasattr(estimator, 'score') and y_true is not None:
        try:
            score_arg_names = _get_arg_names(estimator.score)
            score_args = (X, y_true, sample_weight) if _SAMPLE_WEIGHT in score_arg_names else (X, y_true)
            score = estimator.score(*score_args)
        except Exception as e:
            msg = estimator.score.__qualname__ + " failed. The 'training_score' metric will not be recorded. Scoring error: " + str(e)
            _logger.warning(msg)
        else:
            score_key = prefix + 'score'
            autologging_client.log_metrics(run_id=run_id, metrics={score_key: score})
            metrics[score_key] = score
    _log_estimator_html(run_id, estimator)
    return metrics