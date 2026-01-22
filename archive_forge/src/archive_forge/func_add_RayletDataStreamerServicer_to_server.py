import grpc
from . import ray_client_pb2 as src_dot_ray_dot_protobuf_dot_ray__client__pb2
def add_RayletDataStreamerServicer_to_server(servicer, server):
    rpc_method_handlers = {'Datapath': grpc.stream_stream_rpc_method_handler(servicer.Datapath, request_deserializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.DataRequest.FromString, response_serializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.DataResponse.SerializeToString)}
    generic_handler = grpc.method_handlers_generic_handler('ray.rpc.RayletDataStreamer', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))