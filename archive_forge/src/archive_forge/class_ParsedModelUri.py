import urllib.parse
from typing import NamedTuple, Optional
import mlflow.tracking
from mlflow.exceptions import MlflowException
from mlflow.utils.uri import get_databricks_profile_uri_from_artifact_uri, is_databricks_uri
class ParsedModelUri(NamedTuple):
    name: str
    version: Optional[str] = None
    stage: Optional[str] = None
    alias: Optional[str] = None