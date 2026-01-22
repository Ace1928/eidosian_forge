import grpc
from tensorboard.data.proto import data_provider_pb2 as tensorboard_dot_data_dot_proto_dot_data__provider__pb2
class TensorBoardDataProviderStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetExperiment = channel.unary_unary('/tensorboard.data.TensorBoardDataProvider/GetExperiment', request_serializer=tensorboard_dot_data_dot_proto_dot_data__provider__pb2.GetExperimentRequest.SerializeToString, response_deserializer=tensorboard_dot_data_dot_proto_dot_data__provider__pb2.GetExperimentResponse.FromString)
        self.ListPlugins = channel.unary_unary('/tensorboard.data.TensorBoardDataProvider/ListPlugins', request_serializer=tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ListPluginsRequest.SerializeToString, response_deserializer=tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ListPluginsResponse.FromString)
        self.ListRuns = channel.unary_unary('/tensorboard.data.TensorBoardDataProvider/ListRuns', request_serializer=tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ListRunsRequest.SerializeToString, response_deserializer=tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ListRunsResponse.FromString)
        self.ListScalars = channel.unary_unary('/tensorboard.data.TensorBoardDataProvider/ListScalars', request_serializer=tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ListScalarsRequest.SerializeToString, response_deserializer=tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ListScalarsResponse.FromString)
        self.ReadScalars = channel.unary_unary('/tensorboard.data.TensorBoardDataProvider/ReadScalars', request_serializer=tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ReadScalarsRequest.SerializeToString, response_deserializer=tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ReadScalarsResponse.FromString)
        self.ListTensors = channel.unary_unary('/tensorboard.data.TensorBoardDataProvider/ListTensors', request_serializer=tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ListTensorsRequest.SerializeToString, response_deserializer=tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ListTensorsResponse.FromString)
        self.ReadTensors = channel.unary_unary('/tensorboard.data.TensorBoardDataProvider/ReadTensors', request_serializer=tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ReadTensorsRequest.SerializeToString, response_deserializer=tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ReadTensorsResponse.FromString)
        self.ListBlobSequences = channel.unary_unary('/tensorboard.data.TensorBoardDataProvider/ListBlobSequences', request_serializer=tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ListBlobSequencesRequest.SerializeToString, response_deserializer=tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ListBlobSequencesResponse.FromString)
        self.ReadBlobSequences = channel.unary_unary('/tensorboard.data.TensorBoardDataProvider/ReadBlobSequences', request_serializer=tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ReadBlobSequencesRequest.SerializeToString, response_deserializer=tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ReadBlobSequencesResponse.FromString)
        self.ReadBlob = channel.unary_stream('/tensorboard.data.TensorBoardDataProvider/ReadBlob', request_serializer=tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ReadBlobRequest.SerializeToString, response_deserializer=tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ReadBlobResponse.FromString)