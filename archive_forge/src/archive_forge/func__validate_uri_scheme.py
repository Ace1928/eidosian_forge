import re
from urllib.parse import urlparse, urlunparse
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.http_artifact_repo import HttpArtifactRepository
from mlflow.tracking._tracking_service.utils import get_tracking_uri
def _validate_uri_scheme(scheme):
    allowable_schemes = {'http', 'https'}
    if scheme not in allowable_schemes:
        raise MlflowException(f"The configured tracking uri scheme: '{scheme}' is invalid for use with the proxy mlflow-artifact scheme. The allowed tracking schemes are: {allowable_schemes}")