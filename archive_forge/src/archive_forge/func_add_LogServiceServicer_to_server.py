import grpc
from . import reporter_pb2 as src_dot_ray_dot_protobuf_dot_reporter__pb2
def add_LogServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {'ListLogs': grpc.unary_unary_rpc_method_handler(servicer.ListLogs, request_deserializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.ListLogsRequest.FromString, response_serializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.ListLogsReply.SerializeToString), 'StreamLog': grpc.unary_stream_rpc_method_handler(servicer.StreamLog, request_deserializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.StreamLogRequest.FromString, response_serializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.StreamLogReply.SerializeToString)}
    generic_handler = grpc.method_handlers_generic_handler('ray.rpc.LogService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))