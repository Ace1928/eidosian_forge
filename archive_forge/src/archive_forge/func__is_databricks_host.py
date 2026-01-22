import json
import logging
from typing import Any, Dict, List, Optional
import requests.exceptions
from mlflow import MlflowException
from mlflow.gateway.config import LimitsConfig, Route
from mlflow.gateway.constants import (
from mlflow.gateway.utils import (
from mlflow.protos.databricks_pb2 import BAD_REQUEST
from mlflow.store.entities.paged_list import PagedList
from mlflow.utils.credentials import get_default_host_creds
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.rest_utils import augmented_raise_for_status, http_request
from mlflow.utils.uri import get_uri_scheme
def _is_databricks_host(self) -> bool:
    return self._gateway_uri == 'databricks' or get_uri_scheme(self._gateway_uri) == 'databricks'