import os
import queue
import threading
from typing import Optional
from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import (
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.util import tb_logging
def ActivePlugins(self):
    """Return a set of plugins with summary data.

        Returns:
          The distinct union of `plugin_data.plugin_name` fields from
          all the `SummaryMetadata` protos stored in any run known to
          this multiplexer.
        """
    with self._accumulators_mutex:
        accumulators = list(self._accumulators.values())
    return frozenset().union(*(a.ActivePlugins() for a in accumulators))