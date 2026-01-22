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
@staticmethod
def _get_model_uri_infos(uri):
    from mlflow import MlflowClient
    databricks_profile_uri = get_databricks_profile_uri_from_artifact_uri(uri) or mlflow.get_registry_uri()
    client = MlflowClient(registry_uri=databricks_profile_uri)
    name, version = get_model_name_and_version(client, uri)
    download_uri = client.get_model_version_download_uri(name, version)
    return (name, version, add_databricks_profile_info_to_artifact_uri(download_uri, databricks_profile_uri))