import grpc
from . import gcs_service_pb2 as src_dot_ray_dot_protobuf_dot_gcs__service__pb2
class InternalPubSubGcsServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GcsPublish = channel.unary_unary('/ray.rpc.InternalPubSubGcsService/GcsPublish', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GcsPublishRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GcsPublishReply.FromString)
        self.GcsSubscriberPoll = channel.unary_unary('/ray.rpc.InternalPubSubGcsService/GcsSubscriberPoll', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GcsSubscriberPollRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GcsSubscriberPollReply.FromString)
        self.GcsSubscriberCommandBatch = channel.unary_unary('/ray.rpc.InternalPubSubGcsService/GcsSubscriberCommandBatch', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GcsSubscriberCommandBatchRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GcsSubscriberCommandBatchReply.FromString)