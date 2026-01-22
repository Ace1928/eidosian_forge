from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.entities.run_status import RunStatus
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.protos.service_pb2 import RunInfo as ProtoRunInfo
def _set_run_name(self, new_name):
    self._run_name = new_name