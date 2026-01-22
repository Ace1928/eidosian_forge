import graphene
import mlflow
from mlflow.server.graphql.graphql_custom_scalars import LongString
from mlflow.utils.proto_json_utils import parse_dict
class MlflowRunData(graphene.ObjectType):
    metrics = graphene.List(graphene.NonNull(MlflowMetric))
    params = graphene.List(graphene.NonNull(MlflowParam))
    tags = graphene.List(graphene.NonNull(MlflowRunTag))