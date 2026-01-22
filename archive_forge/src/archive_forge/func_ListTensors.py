import grpc
from tensorboard.data.proto import data_provider_pb2 as tensorboard_dot_data_dot_proto_dot_data__provider__pb2
@staticmethod
def ListTensors(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
    return grpc.experimental.unary_unary(request, target, '/tensorboard.data.TensorBoardDataProvider/ListTensors', tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ListTensorsRequest.SerializeToString, tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ListTensorsResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)