import grpc
from tensorboard.uploader.proto import write_service_pb2 as tensorboard_dot_uploader_dot_proto_dot_write__service__pb2
class TensorBoardWriterService(object):
    """Service for writing data to TensorBoard.dev.
    """

    @staticmethod
    def CreateExperiment(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorboard.service.TensorBoardWriterService/CreateExperiment', tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.CreateExperimentRequest.SerializeToString, tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.CreateExperimentResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def UpdateExperiment(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorboard.service.TensorBoardWriterService/UpdateExperiment', tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.UpdateExperimentRequest.SerializeToString, tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.UpdateExperimentResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeleteExperiment(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorboard.service.TensorBoardWriterService/DeleteExperiment', tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.DeleteExperimentRequest.SerializeToString, tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.DeleteExperimentResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def PurgeData(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorboard.service.TensorBoardWriterService/PurgeData', tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.PurgeDataRequest.SerializeToString, tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.PurgeDataResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def WriteScalar(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorboard.service.TensorBoardWriterService/WriteScalar', tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.WriteScalarRequest.SerializeToString, tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.WriteScalarResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def WriteTensor(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorboard.service.TensorBoardWriterService/WriteTensor', tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.WriteTensorRequest.SerializeToString, tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.WriteTensorResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetOrCreateBlobSequence(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorboard.service.TensorBoardWriterService/GetOrCreateBlobSequence', tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.GetOrCreateBlobSequenceRequest.SerializeToString, tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.GetOrCreateBlobSequenceResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetBlobMetadata(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorboard.service.TensorBoardWriterService/GetBlobMetadata', tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.GetBlobMetadataRequest.SerializeToString, tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.GetBlobMetadataResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def WriteBlob(request_iterator, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/tensorboard.service.TensorBoardWriterService/WriteBlob', tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.WriteBlobRequest.SerializeToString, tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.WriteBlobResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeleteOwnUser(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorboard.service.TensorBoardWriterService/DeleteOwnUser', tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.DeleteOwnUserRequest.SerializeToString, tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.DeleteOwnUserResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)