import collections
import dataclasses
import threading
from typing import Optional
from tensorboard.backend.event_processing import directory_loader
from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing import event_util
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.backend.event_processing import plugin_asset_util
from tensorboard.backend.event_processing import reservoir
from tensorboard.backend.event_processing import tag_types
from tensorboard.compat.proto import config_pb2
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import graph_pb2
from tensorboard.compat.proto import meta_graph_pb2
from tensorboard.compat.proto import tensor_pb2
from tensorboard.util import tb_logging
def _GetPurgeMessage(most_recent_step, most_recent_wall_time, event_step, event_wall_time, num_expired):
    """Return the string message associated with TensorBoard purges."""
    return 'Detected out of order event.step likely caused by a TensorFlow restart. Purging {} expired tensor events from Tensorboard display between the previous step: {} (timestamp: {}) and current step: {} (timestamp: {}).'.format(num_expired, most_recent_step, most_recent_wall_time, event_step, event_wall_time)