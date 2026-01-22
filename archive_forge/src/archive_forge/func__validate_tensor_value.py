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
def _validate_tensor_value(self, tensor_proto, tag, step, wall_time):
    """Validate a TensorProto by attempting to parse it."""
    try:
        tensor_util.make_ndarray(tensor_proto)
    except ValueError as error:
        raise ValueError("The uploader failed to upload a tensor. This seems to be due to a malformation in the tensor, which may be caused by a bug in the process that wrote the tensor.\n\nThe tensor has tag '%s' and is at step %d and wall_time %.6f.\n\nOriginal error:\n%s" % (tag, step, wall_time, error))