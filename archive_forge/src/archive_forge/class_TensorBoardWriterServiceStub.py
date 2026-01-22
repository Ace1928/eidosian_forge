import grpc
from tensorboard.uploader.proto import write_service_pb2 as tensorboard_dot_uploader_dot_proto_dot_write__service__pb2
class TensorBoardWriterServiceStub(object):
    """Service for writing data to TensorBoard.dev.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.CreateExperiment = channel.unary_unary('/tensorboard.service.TensorBoardWriterService/CreateExperiment', request_serializer=tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.CreateExperimentRequest.SerializeToString, response_deserializer=tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.CreateExperimentResponse.FromString)
        self.UpdateExperiment = channel.unary_unary('/tensorboard.service.TensorBoardWriterService/UpdateExperiment', request_serializer=tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.UpdateExperimentRequest.SerializeToString, response_deserializer=tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.UpdateExperimentResponse.FromString)
        self.DeleteExperiment = channel.unary_unary('/tensorboard.service.TensorBoardWriterService/DeleteExperiment', request_serializer=tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.DeleteExperimentRequest.SerializeToString, response_deserializer=tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.DeleteExperimentResponse.FromString)
        self.PurgeData = channel.unary_unary('/tensorboard.service.TensorBoardWriterService/PurgeData', request_serializer=tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.PurgeDataRequest.SerializeToString, response_deserializer=tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.PurgeDataResponse.FromString)
        self.WriteScalar = channel.unary_unary('/tensorboard.service.TensorBoardWriterService/WriteScalar', request_serializer=tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.WriteScalarRequest.SerializeToString, response_deserializer=tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.WriteScalarResponse.FromString)
        self.WriteTensor = channel.unary_unary('/tensorboard.service.TensorBoardWriterService/WriteTensor', request_serializer=tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.WriteTensorRequest.SerializeToString, response_deserializer=tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.WriteTensorResponse.FromString)
        self.GetOrCreateBlobSequence = channel.unary_unary('/tensorboard.service.TensorBoardWriterService/GetOrCreateBlobSequence', request_serializer=tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.GetOrCreateBlobSequenceRequest.SerializeToString, response_deserializer=tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.GetOrCreateBlobSequenceResponse.FromString)
        self.GetBlobMetadata = channel.unary_unary('/tensorboard.service.TensorBoardWriterService/GetBlobMetadata', request_serializer=tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.GetBlobMetadataRequest.SerializeToString, response_deserializer=tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.GetBlobMetadataResponse.FromString)
        self.WriteBlob = channel.stream_stream('/tensorboard.service.TensorBoardWriterService/WriteBlob', request_serializer=tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.WriteBlobRequest.SerializeToString, response_deserializer=tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.WriteBlobResponse.FromString)
        self.DeleteOwnUser = channel.unary_unary('/tensorboard.service.TensorBoardWriterService/DeleteOwnUser', request_serializer=tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.DeleteOwnUserRequest.SerializeToString, response_deserializer=tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.DeleteOwnUserResponse.FromString)