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
def _add_event_internal(self, run_name, event, value, metadata):
    run_proto = self._runs.get(run_name)
    if run_proto is None:
        run_proto = self._create_run(run_name)
        self._runs[run_name] = run_proto
    tag_proto = self._tags.get((run_name, value.tag))
    if tag_proto is None:
        tag_proto = self._create_tag(run_proto, value.tag, metadata)
        self._tags[run_name, value.tag] = tag_proto
    self._create_point(tag_proto, event, value, run_name)
    self._num_values += 1