import os
import queue
import threading
from typing import Optional
from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import (
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.util import tb_logging
def AllSummaryMetadata(self):
    """Return summary metadata for all time series.

        Returns:
          A nested dict `d` such that `d[run][tag]` is a
          `SummaryMetadata` proto for the keyed time series.
        """
    with self._accumulators_mutex:
        items = list(self._accumulators.items())
    return {run_name: accumulator.AllSummaryMetadata() for run_name, accumulator in items}