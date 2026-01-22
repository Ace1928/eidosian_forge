import collections
import threading
import grpc
from grpc import _common
from grpc.beta import _metadata
from grpc.beta import interfaces
from grpc.framework.common import cardinality
from grpc.framework.common import style
from grpc.framework.foundation import abandonment
from grpc.framework.foundation import logging_pool
from grpc.framework.foundation import stream
from grpc.framework.interfaces.face import face
class _ServerProtocolContext(interfaces.GRPCServicerContext):

    def __init__(self, servicer_context):
        self._servicer_context = servicer_context

    def peer(self):
        return self._servicer_context.peer()

    def disable_next_response_compression(self):
        pass