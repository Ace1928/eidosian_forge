import re
from urllib.parse import urlparse, urlunparse
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.http_artifact_repo import HttpArtifactRepository
from mlflow.tracking._tracking_service.utils import get_tracking_uri
def _validate_port_mapped_to_hostname(uri_parse):
    if uri_parse.hostname and _check_if_host_is_numeric(uri_parse.hostname) and (not uri_parse.port):
        raise MlflowException(f'The mlflow-artifacts uri was supplied with a port number: {uri_parse.hostname}, but no host was defined.')