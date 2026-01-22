from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.entities.run_status import RunStatus
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.protos.service_pb2 import RunInfo as ProtoRunInfo
def _copy_with_overrides(self, status=None, end_time=None, lifecycle_stage=None, run_name=None):
    """A copy of the RunInfo with certain attributes modified."""
    proto = self.to_proto()
    if status:
        proto.status = status
    if end_time:
        proto.end_time = end_time
    if lifecycle_stage:
        proto.lifecycle_stage = lifecycle_stage
    if run_name:
        proto.run_name = run_name
    return RunInfo.from_proto(proto)