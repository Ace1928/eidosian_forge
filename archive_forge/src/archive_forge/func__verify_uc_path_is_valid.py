import logging
from typing import Any, Dict
from mlflow.data.dataset_source import DatasetSource
from mlflow.exceptions import MlflowException
def _verify_uc_path_is_valid(self, path):
    """Verify if the path exists in Databricks Unified Catalog."""
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
    except ImportError:
        _logger.warning('Cannot verify the path of `UCVolumeDatasetSource` because of missing`databricks-sdk`. Please install `databricks-sdk` via `pip install -U databricks-sdk`. This does not block creating `UCVolumeDatasetSource`, but your `UCVolumeDatasetSource` might be invalid.')
        return
    except Exception:
        _logger.warning('Cannot verify the path of `UCVolumeDatasetSource` due to a connection failure with Databricks workspace. Please run `mlflow.login()` to log in to Databricks. This does not block creating `UCVolumeDatasetSource`, but your `UCVolumeDatasetSource` might be invalid.')
        return
    try:
        w.files.get_metadata(path)
    except Exception:
        raise MlflowException(f'{path} does not exist in Databricks Unified Catalog.')