import json
import logging
import os
from abc import ABC, abstractmethod
import mlflow
from mlflow.recipes.utils.execution import get_step_output_path
from mlflow.tracking import MlflowClient
from mlflow.tracking._tracking_service.utils import _use_tracking_uri
from mlflow.utils.file_utils import chdir
class DataframeArtifact(Artifact):

    def __init__(self, name, recipe_root, step_name, rel_path=''):
        self._name = name
        self._path = get_step_output_path(recipe_root, step_name, rel_path)
        self._step_name = step_name

    def name(self):
        return self._name

    def path(self):
        return self._path

    def load(self):
        import pandas as pd
        if os.path.exists(self._path):
            return pd.read_parquet(self._path)
        log_artifact_not_found_warning(self._name, self._step_name)
        return None