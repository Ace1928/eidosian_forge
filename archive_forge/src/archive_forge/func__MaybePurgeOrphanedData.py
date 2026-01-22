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
def _MaybePurgeOrphanedData(self, event):
    """Maybe purge orphaned data due to a TensorFlow crash.

        When TensorFlow crashes at step T+O and restarts at step T, any events
        written after step T are now "orphaned" and will be at best misleading if
        they are included in TensorBoard.

        This logic attempts to determine if there is orphaned data, and purge it
        if it is found.

        Args:
          event: The event to use as a reference, to determine if a purge is needed.
        """
    if not self.purge_orphaned_data:
        return
    if self.file_version and self.file_version >= 2:
        self._CheckForRestartAndMaybePurge(event)
    else:
        self._CheckForOutOfOrderStepAndMaybePurge(event)
    if event.HasField('summary'):
        self.most_recent_step = event.step
        self.most_recent_wall_time = event.wall_time