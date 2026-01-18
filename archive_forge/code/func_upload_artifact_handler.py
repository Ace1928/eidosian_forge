import bisect
import json
import logging
import os
import pathlib
import posixpath
import re
import tempfile
import time
import urllib
from functools import wraps
from typing import List, Set
import requests
from flask import Response, current_app, jsonify, request, send_file
from google.protobuf import descriptor
from google.protobuf.json_format import ParseError
from mlflow.entities import DatasetInput, ExperimentTag, FileInfo, Metric, Param, RunTag, ViewType
from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.entities.multipart_upload import MultipartUploadPart
from mlflow.environment_variables import MLFLOW_DEPLOYMENTS_TARGET
from mlflow.exceptions import MlflowException, _UnsupportedMultipartUploadException
from mlflow.models import Model
from mlflow.protos import databricks_pb2
from mlflow.protos.databricks_pb2 import (
from mlflow.protos.mlflow_artifacts_pb2 import (
from mlflow.protos.mlflow_artifacts_pb2 import (
from mlflow.protos.model_registry_pb2 import (
from mlflow.protos.service_pb2 import (
from mlflow.server.validation import _validate_content_type
from mlflow.store.artifact.artifact_repo import MultipartUploadMixin
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.tracking._model_registry import utils as registry_utils
from mlflow.tracking._model_registry.registry import ModelRegistryStoreRegistry
from mlflow.tracking._tracking_service import utils
from mlflow.tracking._tracking_service.registry import TrackingStoreRegistry
from mlflow.tracking.registry import UnsupportedModelRegistryStoreURIException
from mlflow.utils.file_utils import local_file_uri_to_path
from mlflow.utils.mime_type_utils import _guess_mime_type
from mlflow.utils.promptlab_utils import _create_promptlab_run_impl
from mlflow.utils.proto_json_utils import message_to_json, parse_dict
from mlflow.utils.string_utils import is_string_type
from mlflow.utils.uri import is_local_uri, validate_path_is_safe, validate_query_string
from mlflow.utils.validation import _validate_batch_log_api_req
@catch_mlflow_exception
def upload_artifact_handler():
    args = request.args
    run_uuid = args.get('run_uuid')
    if not run_uuid:
        raise MlflowException(message='Request must specify run_uuid.', error_code=INVALID_PARAMETER_VALUE)
    path = args.get('path')
    if not path:
        raise MlflowException(message='Request must specify path.', error_code=INVALID_PARAMETER_VALUE)
    path = validate_path_is_safe(path)
    if request.content_length and request.content_length > 10 * 1024 * 1024:
        raise MlflowException(message='Artifact size is too large. Max size is 10MB.', error_code=INVALID_PARAMETER_VALUE)
    data = request.data
    if not data:
        raise MlflowException(message='Request must specify data.', error_code=INVALID_PARAMETER_VALUE)
    run = _get_tracking_store().get_run(run_uuid)
    artifact_dir = run.info.artifact_uri
    basename = posixpath.basename(path)
    dirname = posixpath.dirname(path)

    def _log_artifact_to_repo(file, run, dirname, artifact_dir):
        if _is_servable_proxied_run_artifact_root(run.info.artifact_uri):
            artifact_repo = _get_artifact_repo_mlflow_artifacts()
            path_to_log = os.path.join(run.info.experiment_id, run.info.run_id, 'artifacts', dirname) if dirname else os.path.join(run.info.experiment_id, run.info.run_id, 'artifacts')
        else:
            artifact_repo = get_artifact_repository(artifact_dir)
            path_to_log = dirname
        artifact_repo.log_artifact(file, path_to_log)
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = os.path.join(tmpdir, dirname) if dirname else tmpdir
        file_path = os.path.join(dir_path, basename)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            f.write(data)
        _log_artifact_to_repo(file_path, run, dirname, artifact_dir)
    return Response(mimetype='application/json')