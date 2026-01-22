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
class _Server(interfaces.Server):

    def __init__(self, grpc_server):
        self._grpc_server = grpc_server

    def add_insecure_port(self, address):
        return self._grpc_server.add_insecure_port(address)

    def add_secure_port(self, address, server_credentials):
        return self._grpc_server.add_secure_port(address, server_credentials)

    def start(self):
        self._grpc_server.start()

    def stop(self, grace):
        return self._grpc_server.stop(grace)

    def __enter__(self):
        self._grpc_server.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._grpc_server.stop(None)
        return False