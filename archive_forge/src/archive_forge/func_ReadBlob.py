import grpc
from tensorboard.data.proto import data_provider_pb2 as tensorboard_dot_data_dot_proto_dot_data__provider__pb2
@staticmethod
def ReadBlob(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
    return grpc.experimental.unary_stream(request, target, '/tensorboard.data.TensorBoardDataProvider/ReadBlob', tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ReadBlobRequest.SerializeToString, tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ReadBlobResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)