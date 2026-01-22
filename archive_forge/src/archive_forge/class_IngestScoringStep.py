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
class IngestScoringStep(BaseIngestStep):
    _DATASET_OUTPUT_NAME = 'scoring-dataset.parquet'

    def __init__(self, step_config: Dict[str, Any], recipe_root: str):
        super().__init__(step_config, recipe_root)
        self.dataset_output_name = IngestScoringStep._DATASET_OUTPUT_NAME

    @classmethod
    def from_recipe_config(cls, recipe_config: Dict[str, Any], recipe_root: str):
        step_config = recipe_config.get('steps', {}).get('ingest_scoring', {})
        step_config['recipe'] = recipe_config.get('recipe')
        return cls(step_config=step_config, recipe_root=recipe_root)

    @property
    def name(self) -> str:
        return 'ingest_scoring'

    def get_artifacts(self):
        return [DataframeArtifact('ingested_scoring_data', self.recipe_root, self.name, IngestScoringStep._DATASET_OUTPUT_NAME)]

    def step_class(self):
        return StepClass.PREDICTION