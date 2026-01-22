import grpc
from tensorboard.uploader.proto import export_service_pb2 as tensorboard_dot_uploader_dot_proto_dot_export__service__pb2
class TensorBoardExporterService(object):
    """Service for exporting data from TensorBoard.dev.
    """

    @staticmethod
    def StreamExperiments(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_stream(request, target, '/tensorboard.service.TensorBoardExporterService/StreamExperiments', tensorboard_dot_uploader_dot_proto_dot_export__service__pb2.StreamExperimentsRequest.SerializeToString, tensorboard_dot_uploader_dot_proto_dot_export__service__pb2.StreamExperimentsResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def StreamExperimentData(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_stream(request, target, '/tensorboard.service.TensorBoardExporterService/StreamExperimentData', tensorboard_dot_uploader_dot_proto_dot_export__service__pb2.StreamExperimentDataRequest.SerializeToString, tensorboard_dot_uploader_dot_proto_dot_export__service__pb2.StreamExperimentDataResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def StreamBlobData(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_stream(request, target, '/tensorboard.service.TensorBoardExporterService/StreamBlobData', tensorboard_dot_uploader_dot_proto_dot_export__service__pb2.StreamBlobDataRequest.SerializeToString, tensorboard_dot_uploader_dot_proto_dot_export__service__pb2.StreamBlobDataResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)