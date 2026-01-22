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
def _get_request_message(request_message, flask_request=request, schema=None):
    from querystring_parser import parser
    if flask_request.method == 'GET' and len(flask_request.query_string) > 0:
        query_string = re.sub('%5B%5D', '%5B0%5D', flask_request.query_string.decode('utf-8'))
        request_dict = parser.parse(query_string, normalized=True)
        for field in request_message.DESCRIPTOR.fields:
            if field.label == descriptor.FieldDescriptor.LABEL_REPEATED and field.name in request_dict:
                if not isinstance(request_dict[field.name], list):
                    request_dict[field.name] = [request_dict[field.name]]
        request_json = request_dict
    else:
        request_json = _get_request_json(flask_request)
        if is_string_type(request_json):
            request_json = json.loads(request_json)
        if request_json is None:
            request_json = {}
    proto_parsing_succeeded = True
    try:
        parse_dict(request_json, request_message)
    except ParseError:
        proto_parsing_succeeded = False
    schema = schema or {}
    for schema_key, schema_validation_fns in schema.items():
        if schema_key in request_json or _assert_required in schema_validation_fns:
            value = request_json.get(schema_key)
            if schema_key == 'run_id' and value is None and ('run_uuid' in request_json):
                value = request_json.get('run_uuid')
            _validate_param_against_schema(schema=schema_validation_fns, param=schema_key, value=value, proto_parsing_succeeded=proto_parsing_succeeded)
    return request_message