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
@_disable_if_artifacts_only
def _list_artifacts():
    request_message = _get_request_message(ListArtifacts(), schema={'run_id': [_assert_string, _assert_required], 'path': [_assert_string], 'page_token': [_assert_string]})
    response_message = ListArtifacts.Response()
    if request_message.HasField('path'):
        path = request_message.path
        path = validate_path_is_safe(path)
    else:
        path = None
    run_id = request_message.run_id or request_message.run_uuid
    run = _get_tracking_store().get_run(run_id)
    if _is_servable_proxied_run_artifact_root(run.info.artifact_uri):
        artifact_entities = _list_artifacts_for_proxied_run_artifact_root(proxied_artifact_root=run.info.artifact_uri, relative_path=path)
    else:
        artifact_entities = _get_artifact_repo(run).list_artifacts(path)
    response_message.files.extend([a.to_proto() for a in artifact_entities])
    response_message.root_uri = run.info.artifact_uri
    response = Response(mimetype='application/json')
    response.set_data(message_to_json(response_message))
    return response