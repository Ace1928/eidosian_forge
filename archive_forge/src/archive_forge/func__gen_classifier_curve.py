import copy
import functools
import inspect
import json
import logging
import math
import pathlib
import pickle
import shutil
import tempfile
import time
import traceback
import warnings
from collections import namedtuple
from functools import partial
from typing import Callable, List, NamedTuple, Optional, Tuple, Union
import numpy as np
import pandas as pd
from packaging.version import Version
from sklearn import metrics as sk_metrics
from sklearn.pipeline import Pipeline as sk_Pipeline
import mlflow
from mlflow import MlflowClient
from mlflow.entities.metric import Metric
from mlflow.exceptions import MlflowException
from mlflow.metrics import (
from mlflow.models.evaluation.artifacts import (
from mlflow.models.evaluation.base import (
from mlflow.models.utils import plot_lines
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.pyfunc import _ServedPyFuncModel
from mlflow.sklearn import _SklearnModelWrapper
from mlflow.utils.file_utils import TempDir
from mlflow.utils.proto_json_utils import NumpyEncoder
from mlflow.utils.time import get_current_time_millis
def _gen_classifier_curve(is_binomial, y, y_probs, labels, pos_label, curve_type, sample_weights):
    """
    Generate precision-recall curve or ROC curve for classifier.

    Args:
        is_binomial: True if it is binary classifier otherwise False
        y: True label values
        y_probs: if binary classifier, the predicted probability for positive class.
                  if multiclass classifier, the predicted probabilities for all classes.
        labels: The set of labels.
        pos_label: The label of the positive class.
        curve_type: "pr" or "roc"
        sample_weights: Optional sample weights.

    Returns:
        An instance of "_Curve" which includes attributes "plot_fn", "plot_fn_args", "auc".
    """
    if curve_type == 'roc':

        def gen_line_x_y_label_auc(_y, _y_prob, _pos_label):
            fpr, tpr, _ = sk_metrics.roc_curve(_y, _y_prob, sample_weight=sample_weights, pos_label=_pos_label if _pos_label == pos_label else None)
            auc = sk_metrics.roc_auc_score(y_true=_y, y_score=_y_prob, sample_weight=sample_weights)
            return (fpr, tpr, f'AUC={auc:.3f}', auc)
        xlabel = 'False Positive Rate'
        ylabel = 'True Positive Rate'
        title = 'ROC curve'
        if pos_label:
            xlabel = f'False Positive Rate (Positive label: {pos_label})'
            ylabel = f'True Positive Rate (Positive label: {pos_label})'
    elif curve_type == 'pr':

        def gen_line_x_y_label_auc(_y, _y_prob, _pos_label):
            precision, recall, _ = sk_metrics.precision_recall_curve(_y, _y_prob, sample_weight=sample_weights, pos_label=_pos_label if _pos_label == pos_label else None)
            ap = sk_metrics.average_precision_score(y_true=_y, y_score=_y_prob, pos_label=_pos_label, sample_weight=sample_weights)
            return (recall, precision, f'AP={ap:.3f}', ap)
        xlabel = 'Recall'
        ylabel = 'Precision'
        title = 'Precision recall curve'
        if pos_label:
            xlabel = f'Recall (Positive label: {pos_label})'
            ylabel = f'Precision (Positive label: {pos_label})'
    else:
        assert False, 'illegal curve type'
    if is_binomial:
        x_data, y_data, line_label, auc = gen_line_x_y_label_auc(y, y_probs, pos_label)
        data_series = [(line_label, x_data, y_data)]
    else:
        curve_list = []
        for positive_class_index, positive_class in enumerate(labels):
            y_bin, _, y_prob_bin = _get_binary_sum_up_label_pred_prob(positive_class_index, positive_class, y, labels, y_probs)
            x_data, y_data, line_label, auc = gen_line_x_y_label_auc(y_bin, y_prob_bin, _pos_label=1)
            curve_list.append((positive_class, x_data, y_data, line_label, auc))
        data_series = [(f'label={positive_class},{line_label}', x_data, y_data) for positive_class, x_data, y_data, line_label, _ in curve_list]
        auc = [auc for _, _, _, _, auc in curve_list]

    def _do_plot(**kwargs):
        from matplotlib import pyplot
        _, ax = plot_lines(**kwargs)
        dash_line_args = {'color': 'gray', 'alpha': 0.3, 'drawstyle': 'default', 'linestyle': 'dashed'}
        if curve_type == 'pr':
            ax.plot([0, 1], [1, 0], **dash_line_args)
        elif curve_type == 'roc':
            ax.plot([0, 1], [0, 1], **dash_line_args)
        if is_binomial:
            ax.legend(loc='best')
        else:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            pyplot.subplots_adjust(right=0.6, bottom=0.25)
    return _Curve(plot_fn=_do_plot, plot_fn_args={'data_series': data_series, 'xlabel': xlabel, 'ylabel': ylabel, 'line_kwargs': {'drawstyle': 'steps-post', 'linewidth': 1}, 'title': title}, auc=auc)