from dataclasses import dataclass
from mlflow import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
def _validate_permission(permission: str):
    if permission not in ALL_PERMISSIONS:
        raise MlflowException(f"Invalid permission '{permission}'. Valid permissions are: {tuple(ALL_PERMISSIONS)}", INVALID_PARAMETER_VALUE)