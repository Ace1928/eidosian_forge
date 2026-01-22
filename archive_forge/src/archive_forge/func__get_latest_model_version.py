import urllib.parse
from typing import NamedTuple, Optional
import mlflow.tracking
from mlflow.exceptions import MlflowException
from mlflow.utils.uri import get_databricks_profile_uri_from_artifact_uri, is_databricks_uri
def _get_latest_model_version(client, name, stage):
    """
    Returns the latest version of the stage if stage is not None. Otherwise return the latest of all
    versions.
    """
    latest = client.get_latest_versions(name, None if stage is None else [stage])
    if len(latest) == 0:
        stage_str = '' if stage is None else f" and stage '{stage}'"
        raise MlflowException(f"No versions of model with name '{name}'{stage_str} found")
    return max((int(x.version) for x in latest))