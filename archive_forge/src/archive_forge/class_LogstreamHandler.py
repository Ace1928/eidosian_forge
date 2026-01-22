import io
import logging
import queue
import threading
import uuid
import grpc
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray._private.ray_logging import global_worker_stdstream_dispatcher
from ray._private.worker import print_worker_logs
from ray.util.client.common import CLIENT_SERVER_MAX_THREADS
class LogstreamHandler(logging.Handler):

    def __init__(self, queue, level):
        super().__init__()
        self.queue = queue
        self.level = level

    def emit(self, record: logging.LogRecord):
        logdata = ray_client_pb2.LogData()
        logdata.msg = record.getMessage()
        logdata.level = record.levelno
        logdata.name = record.name
        self.queue.put(logdata)