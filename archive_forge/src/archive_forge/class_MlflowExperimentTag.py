import graphene
import mlflow
from mlflow.server.graphql.graphql_custom_scalars import LongString
from mlflow.utils.proto_json_utils import parse_dict
class MlflowExperimentTag(graphene.ObjectType):
    key = graphene.String()
    value = graphene.String()