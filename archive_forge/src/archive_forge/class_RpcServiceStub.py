import grpc
from tensorflow.distribute.experimental.rpc.proto import tf_rpc_service_pb2 as tensorflow_dot_distribute_dot_experimental_dot_rpc_dot_proto_dot_tf__rpc__service__pb2
class RpcServiceStub(object):
    pass

    def __init__(self, channel):
        """Constructor.

    Args:
      channel: A grpc.Channel.
    """
        self.Call = channel.unary_unary('/tensorflow.rpc.RpcService/Call', request_serializer=tensorflow_dot_distribute_dot_experimental_dot_rpc_dot_proto_dot_tf__rpc__service__pb2.CallRequest.SerializeToString, response_deserializer=tensorflow_dot_distribute_dot_experimental_dot_rpc_dot_proto_dot_tf__rpc__service__pb2.CallResponse.FromString)
        self.List = channel.unary_unary('/tensorflow.rpc.RpcService/List', request_serializer=tensorflow_dot_distribute_dot_experimental_dot_rpc_dot_proto_dot_tf__rpc__service__pb2.ListRequest.SerializeToString, response_deserializer=tensorflow_dot_distribute_dot_experimental_dot_rpc_dot_proto_dot_tf__rpc__service__pb2.ListResponse.FromString)