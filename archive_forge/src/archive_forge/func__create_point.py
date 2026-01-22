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
def _create_point(self, tag_proto, event, value, run_name):
    """Adds a tensor point to the given tag, if there's space.

        Args:
          tag_proto: `WriteTensorRequest.Tag` proto to which to add a point.
          event: Enclosing `Event` proto with the step and wall time data.
          value: Tensor `Summary.Value` proto with the actual tensor data.
          run_name: Name of the wrong, only used for error reporting.

        Raises:
          _OutOfSpaceError: If adding the point would exceed the remaining
            request budget.
        """
    point = tag_proto.points.add()
    point.step = event.step
    point.value.CopyFrom(value.tensor)
    util.set_timestamp(point.wall_time, event.wall_time)
    self._tensor_bytes += point.value.ByteSize()
    if point.value.ByteSize() > self._max_tensor_point_size:
        logger.warning('Tensor (run:%s, tag:%s, step: %d) too large; skipping. Size %d exceeds limit of %d bytes.', run_name, tag_proto.name, event.step, point.value.ByteSize(), self._max_tensor_point_size)
        tag_proto.points.pop()
        self._num_values_skipped += 1
        self._tensor_bytes_skipped += point.value.ByteSize()
        return
    self._validate_tensor_value(value.tensor, value.tag, event.step, event.wall_time)
    try:
        self._byte_budget_manager.add_point(point)
    except _OutOfSpaceError:
        tag_proto.points.pop()
        raise