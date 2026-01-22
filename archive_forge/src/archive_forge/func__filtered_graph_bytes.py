import contextlib
import functools
import time
import grpc
from google.protobuf import message
from tensorboard.compat.proto import graph_pb2
from tensorboard.compat.proto import summary_pb2
from tensorboard.compat.proto import types_pb2
from tensorboard.uploader.proto import write_service_pb2
from tensorboard.uploader import logdir_loader
from tensorboard.uploader import upload_tracker
from tensorboard.uploader import util
from tensorboard.backend import process_graph
from tensorboard.backend.event_processing import directory_loader
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.plugins.graph import metadata as graphs_metadata
from tensorboard.util import grpc_util
from tensorboard.util import tb_logging
from tensorboard.util import tensor_util
def _filtered_graph_bytes(graph_bytes):
    try:
        graph_def = graph_pb2.GraphDef().FromString(graph_bytes)
    except (message.DecodeError, RuntimeWarning):
        logger.warning('Could not parse GraphDef of size %d. Skipping.', len(graph_bytes))
        return None
    process_graph.prepare_graph_for_ui(graph_def)
    return graph_def.SerializeToString()