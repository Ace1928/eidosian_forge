import grpc
from . import job_agent_pb2 as src_dot_ray_dot_protobuf_dot_job__agent__pb2
class JobAgentServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.InitializeJobEnv = channel.unary_unary('/ray.rpc.JobAgentService/InitializeJobEnv', request_serializer=src_dot_ray_dot_protobuf_dot_job__agent__pb2.InitializeJobEnvRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_job__agent__pb2.InitializeJobEnvReply.FromString)