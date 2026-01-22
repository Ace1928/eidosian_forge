import logging
import os
import urllib.parse
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.databricks_models_artifact_repo import DatabricksModelsArtifactRepository
from mlflow.store.artifact.unity_catalog_models_artifact_repo import (
from mlflow.store.artifact.utils.models import (
from mlflow.utils.file_utils import write_yaml
from mlflow.utils.uri import (
def _add_registered_model_meta_file(self, model_path):
    write_yaml(model_path, REGISTERED_MODEL_META_FILE_NAME, {'model_name': self.model_name, 'model_version': self.model_version}, overwrite=True, ensure_yaml_extension=False)