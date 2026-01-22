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
def Tags(self):
    """Return all tags found in the value stream.

        Returns:
          A `{tagType: ['list', 'of', 'tags']}` dictionary.
        """
    return {TENSORS: list(self.tensors_by_tag.keys()), GRAPH: self._graph is not None, META_GRAPH: self._meta_graph is not None, RUN_METADATA: list(self._tagged_metadata.keys())}