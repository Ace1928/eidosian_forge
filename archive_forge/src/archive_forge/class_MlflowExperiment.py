import graphene
import mlflow
from mlflow.server.graphql.graphql_custom_scalars import LongString
from mlflow.utils.proto_json_utils import parse_dict
class MlflowExperiment(graphene.ObjectType):
    experiment_id = graphene.String()
    name = graphene.String()
    artifact_location = graphene.String()
    lifecycle_stage = graphene.String()
    last_update_time = LongString()
    creation_time = LongString()
    tags = graphene.List(graphene.NonNull(MlflowExperimentTag))