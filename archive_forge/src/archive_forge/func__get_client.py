import logging
import os
from packaging import version
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Optional
from ray._private.dict import flatten_dict
def _get_client(self) -> 'MlflowClient':
    """Returns an ml.tracking.MlflowClient instance to use for logging."""
    tracking_uri = self._mlflow.get_tracking_uri()
    registry_uri = self._mlflow.get_registry_uri()
    from mlflow.tracking import MlflowClient
    return MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)