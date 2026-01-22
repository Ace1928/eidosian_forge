from __future__ import absolute_import, division, print_function
import sys
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import enum_type_wrapper
import grpc
from grpc.beta import implementations as beta_implementations
from grpc.beta import interfaces as beta_interfaces
from grpc.framework.common import cardinality
from grpc.framework.interfaces.face import utilities as face_utilities
def add_gRPCExecServicer_to_server(servicer, server):
    rpc_method_handlers = {'ShowCmdTextOutput': grpc.unary_stream_rpc_method_handler(servicer.ShowCmdTextOutput, request_deserializer=ShowCmdArgs.FromString, response_serializer=ShowCmdTextReply.SerializeToString), 'ShowCmdJSONOutput': grpc.unary_stream_rpc_method_handler(servicer.ShowCmdJSONOutput, request_deserializer=ShowCmdArgs.FromString, response_serializer=ShowCmdJSONReply.SerializeToString)}
    generic_handler = grpc.method_handlers_generic_handler('IOSXRExtensibleManagabilityService.gRPCExec', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))