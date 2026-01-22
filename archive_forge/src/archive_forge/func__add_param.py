from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.metric import Metric
from mlflow.entities.param import Param
from mlflow.entities.run_tag import RunTag
from mlflow.protos.service_pb2 import Param as ProtoParam
from mlflow.protos.service_pb2 import RunData as ProtoRunData
from mlflow.protos.service_pb2 import RunTag as ProtoRunTag
def _add_param(self, param):
    self._params[param.key] = param.value