import base64
import functools
import logging
import os
import shutil
from contextlib import contextmanager
import mlflow
from mlflow.entities import Run
from mlflow.environment_variables import MLFLOW_UNITY_CATALOG_PRESIGNED_URLS_ENABLED
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
from mlflow.protos.databricks_uc_registry_service_pb2 import UcModelRegistryService
from mlflow.protos.service_pb2 import GetRun, MlflowService
from mlflow.store.artifact.presigned_url_artifact_repo import PresignedUrlArtifactRepository
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry.rest_store import BaseRestStore
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils._unity_catalog_utils import (
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import get_databricks_host_creds, is_databricks_uri
from mlflow.utils.mlflow_tags import (
from mlflow.utils.proto_json_utils import message_to_json, parse_dict
from mlflow.utils.rest_utils import (
def _download_model_weights_if_not_saved(self, local_model_path):
    """
        Transformers models can be saved without the base model weights by setting
        `save_pretrained=False` when saving or logging the model. Such 'weight-less'
        model cannot be directly deployed to model serving, so here we download the
        weights proactively from the HuggingFace hub and save them to the model directory.
        """
    model = _load_model(local_model_path)
    flavor_conf = model.flavors.get('transformers')
    if not flavor_conf:
        return
    from mlflow.transformers.flavor_config import FlavorKey
    from mlflow.transformers.model_io import _MODEL_BINARY_FILE_NAME
    if FlavorKey.MODEL_BINARY in flavor_conf and os.path.exists(os.path.join(local_model_path, _MODEL_BINARY_FILE_NAME)) and (FlavorKey.MODEL_REVISION not in flavor_conf):
        return
    _logger.info('You are attempting to register a transformers model that does not have persisted model weights. Attempting to fetch the weights so that the model can be registered within Unity Catalog.')
    try:
        mlflow.transformers.persist_pretrained_model(local_model_path)
    except Exception as e:
        raise MlflowException('Failed to download the model weights from the HuggingFace hub and cannot register the model in the Unity Catalog. Please ensure that the model was saved with the correct reference to the HuggingFace hub repository and that you have access to fetch model weights from the defined repository.', error_code=INTERNAL_ERROR) from e