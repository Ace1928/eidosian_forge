import graphene
import mlflow
from mlflow.server.graphql.graphql_custom_scalars import LongString
from mlflow.utils.proto_json_utils import parse_dict
class QueryType(graphene.ObjectType):
    mlflow_get_experiment = graphene.Field(MlflowGetExperimentResponse, input=MlflowGetExperimentInput())
    mlflow_get_run = graphene.Field(MlflowGetRunResponse, input=MlflowGetRunInput())
    mlflow_search_model_versions = graphene.Field(MlflowSearchModelVersionsResponse, input=MlflowSearchModelVersionsInput())

    def resolve_mlflow_get_experiment(self, info, input):
        input_dict = vars(input)
        request_message = mlflow.protos.service_pb2.GetExperiment()
        parse_dict(input_dict, request_message)
        return mlflow.server.handlers.get_experiment_impl(request_message)

    def resolve_mlflow_get_run(self, info, input):
        input_dict = vars(input)
        request_message = mlflow.protos.service_pb2.GetRun()
        parse_dict(input_dict, request_message)
        return mlflow.server.handlers.get_run_impl(request_message)

    def resolve_mlflow_search_model_versions(self, info, input):
        input_dict = vars(input)
        request_message = mlflow.protos.model_registry_pb2.SearchModelVersions()
        parse_dict(input_dict, request_message)
        return mlflow.server.handlers.search_model_versions_impl(request_message)