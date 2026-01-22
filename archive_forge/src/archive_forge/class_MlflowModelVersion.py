import graphene
import mlflow
from mlflow.server.graphql.graphql_custom_scalars import LongString
from mlflow.utils.proto_json_utils import parse_dict
class MlflowModelVersion(graphene.ObjectType):
    name = graphene.String()
    version = graphene.String()
    creation_timestamp = LongString()
    last_updated_timestamp = LongString()
    user_id = graphene.String()
    current_stage = graphene.String()
    description = graphene.String()
    source = graphene.String()
    run_id = graphene.String()
    status = graphene.Field(MlflowModelVersionStatus)
    status_message = graphene.String()
    tags = graphene.List(graphene.NonNull(MlflowModelVersionTag))
    run_link = graphene.String()
    aliases = graphene.List(graphene.String)