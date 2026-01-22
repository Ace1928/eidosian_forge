import graphene
import mlflow
from mlflow.server.graphql.graphql_custom_scalars import LongString
from mlflow.utils.proto_json_utils import parse_dict
class MlflowModelVersionStatus(graphene.Enum):
    PENDING_REGISTRATION = 1
    FAILED_REGISTRATION = 2
    READY = 3