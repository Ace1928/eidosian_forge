import grpc
from tensorflow.distribute.experimental.rpc.proto import tf_rpc_service_pb2 as tensorflow_dot_distribute_dot_experimental_dot_rpc_dot_proto_dot_tf__rpc__service__pb2
def Call(self, request, context):
    """RPC for invoking a registered function on remote server.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')