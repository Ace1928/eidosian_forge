import collections
import dataclasses
import threading
from typing import Optional, Sequence, Tuple
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
from tensorboard.plugins.distribution import compressor
from tensorboard.util import tb_logging
@dataclasses.dataclass(frozen=True)
class ImageEvent:
    """Contains information of an image event.

    Attributes:
      wall_time: Timestamp of the event in seconds.
      step: Global step of the event.
      encoded_image_string: Image content encoded in bytes.
      width: Width of the image.
      height: Height of the image.
    """
    wall_time: float
    step: int
    encoded_image_string: bytes
    width: int
    height: int