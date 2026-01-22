import grpc
from . import job_agent_pb2 as src_dot_ray_dot_protobuf_dot_job__agent__pb2
class JobAgentServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def InitializeJobEnv(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')