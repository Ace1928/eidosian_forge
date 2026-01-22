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
def _CheckForRestartAndMaybePurge(self, event):
    """Check and discard expired events using SessionLog.START.

        The first SessionLog.START event in a run indicates the start of a
        supervisor session. Subsequent SessionLog.START events indicate a
        *restart*, which may need to preempt old events. This method checks
        for a session restart event and purges all previously seen events whose
        step is larger than or equal to this event's step.

        Because of supervisor threading, it is possible that this logic will
        cause the first few event messages to be discarded since supervisor
        threading does not guarantee that the START message is deterministically
        written first.

        This method is preferred over _CheckForOutOfOrderStepAndMaybePurge which
        can inadvertently discard events due to supervisor threading.

        Args:
          event: The event to use as reference. If the event is a START event, all
            previously seen events with a greater event.step will be purged.
        """
    if event.session_log.status != event_pb2.SessionLog.START:
        return
    if not self._seen_session_start:
        self._seen_session_start = True
        return
    self._Purge(event, by_tags=False)