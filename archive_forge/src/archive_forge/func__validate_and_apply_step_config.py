import abc
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.recipes.artifacts import DataframeArtifact
from mlflow.recipes.cards import BaseCard
from mlflow.recipes.step import BaseStep, StepClass
from mlflow.recipes.steps.ingest.datasets import (
from mlflow.recipes.utils.step import get_pandas_data_profiles, validate_classification_config
from mlflow.utils.file_utils import read_parquet_as_pandas_df
def _validate_and_apply_step_config(self):
    dataset_format = self.step_config.get('using')
    if not dataset_format:
        raise MlflowException(message='Dataset format must be specified via the `using` key within the `ingest` section of recipe.yaml', error_code=INVALID_PARAMETER_VALUE)
    if self.step_class() == StepClass.TRAINING:
        self.target_col = self.step_config.get('target_col')
        if self.target_col is None:
            raise MlflowException('Missing target_col config in recipe config.', error_code=INVALID_PARAMETER_VALUE)
        self.positive_class = self.step_config.get('positive_class')
    for dataset_class in BaseIngestStep._SUPPORTED_DATASETS:
        if dataset_class.handles_format(dataset_format):
            self.dataset = dataset_class.from_config(dataset_config=self.step_config, recipe_root=self.recipe_root)
            break
    else:
        raise MlflowException(message=f'Unrecognized dataset format: {dataset_format}', error_code=INVALID_PARAMETER_VALUE)
    self.skip_data_profiling = self.step_config.get('skip_data_profiling', False)