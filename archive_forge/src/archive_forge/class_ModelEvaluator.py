import json
import keyword
import logging
import math
import operator
import os
import pathlib
import signal
import struct
import sys
import urllib
import urllib.parse
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from decimal import Decimal
from types import FunctionType
from typing import Any, Dict, Optional
import mlflow
from mlflow.data.dataset import Dataset
from mlflow.entities import RunTag
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation.validation import (
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.client import MlflowClient
from mlflow.utils import _get_fully_qualified_class_name, insecure_hash
from mlflow.utils.annotations import developer_stable, experimental
from mlflow.utils.class_utils import _get_class_from_string
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import MLFLOW_DATASET_CONTEXT
from mlflow.utils.proto_json_utils import NumpyEncoder
from mlflow.utils.string_utils import generate_feature_name_if_not_string
@developer_stable
class ModelEvaluator(metaclass=ABCMeta):

    @abstractmethod
    def can_evaluate(self, *, model_type, evaluator_config, **kwargs) -> bool:
        """
        Args:
            model_type: A string describing the model type (e.g., "regressor", "classifier", …).
            evaluator_config: A dictionary of additional configurations for
                the evaluator.
            kwargs: For forwards compatibility, a placeholder for additional arguments
                that may be added to the evaluation interface in the future.

        Returns:
            True if the evaluator can evaluate the specified model on the
            specified dataset. False otherwise.
        """
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, *, model_type, dataset, run_id, evaluator_config, model=None, custom_metrics=None, extra_metrics=None, custom_artifacts=None, baseline_model=None, predictions=None, **kwargs):
        """
        The abstract API to log metrics and artifacts, and return evaluation results.

        Args:
            model_type: A string describing the model type
                (e.g., ``"regressor"``, ``"classifier"``, …).
            dataset: An instance of `mlflow.models.evaluation.base._EvaluationDataset`
                containing features and labels (optional) for model evaluation.
            run_id: The ID of the MLflow Run to which to log results.
            evaluator_config: A dictionary of additional configurations for
                the evaluator.
            model: A pyfunc model instance, used as the candidate_model
                to be compared with baseline_model (specified by the `baseline_model` param)
                for model validation. If None, the model output is supposed to be found in
                ``dataset.predictions_data``.
            extra_metrics: A list of :py:class:`EvaluationMetric` objects.
            custom_artifacts: A list of callable custom artifact functions.
            kwargs: For forwards compatibility, a placeholder for additional arguments that
                may be added to the evaluation interface in the future.
            baseline_model: (Optional) A string URI referring to a MLflow model with the pyfunc
                flavor as a baseline model to be compared with the
                candidate model (specified by the `model` param) for model
                validation. (pyfunc model instance is not allowed)
            predictions: The column name of the model output column that is used for evaluation.
                This is only used when a model returns a pandas dataframe that contains
                multiple columns.

        Returns:
            A :py:class:`mlflow.models.EvaluationResult` instance containing
            evaluation metrics for candidate model and baseline model and
            artifacts for candidate model.
        """
        raise NotImplementedError()