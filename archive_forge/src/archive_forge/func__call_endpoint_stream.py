import json
import posixpath
from typing import Any, Dict, Iterator, Optional
from mlflow.deployments import BaseDeploymentClient
from mlflow.deployments.constants import (
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.utils import AttrDict
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.rest_utils import augmented_raise_for_status, http_request
def _call_endpoint_stream(self, *, method: str, prefix: str='/api/2.0', route: Optional[str]=None, json_body: Optional[Dict[str, Any]]=None, timeout: Optional[int]=None) -> Iterator[str]:
    call_kwargs = {}
    if method.lower() == 'get':
        call_kwargs['params'] = json_body
    else:
        call_kwargs['json'] = json_body
    response = http_request(host_creds=get_databricks_host_creds(self.target_uri), endpoint=posixpath.join(prefix, 'serving-endpoints', route or ''), method=method, timeout=MLFLOW_HTTP_REQUEST_TIMEOUT.get() if timeout is None else timeout, raise_on_status=False, retry_codes=MLFLOW_DEPLOYMENT_CLIENT_REQUEST_RETRY_CODES, extra_headers={'X-Databricks-Endpoints-API-Client': 'Databricks Deployment Client'}, stream=True, **call_kwargs)
    augmented_raise_for_status(response)
    return (line.strip() for line in response.iter_lines(decode_unicode=True) if line.strip())