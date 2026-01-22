import grpc
from . import core_worker_pb2 as src_dot_ray_dot_protobuf_dot_core__worker__pb2
from . import pubsub_pb2 as src_dot_ray_dot_protobuf_dot_pubsub__pb2
@staticmethod
def WaitForActorOutOfScope(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
    return grpc.experimental.unary_unary(request, target, '/ray.rpc.CoreWorkerService/WaitForActorOutOfScope', src_dot_ray_dot_protobuf_dot_core__worker__pb2.WaitForActorOutOfScopeRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_core__worker__pb2.WaitForActorOutOfScopeReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)