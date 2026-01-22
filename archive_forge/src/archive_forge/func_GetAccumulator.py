import os
import threading
from typing import Optional
from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.util import tb_logging
def GetAccumulator(self, run):
    """Returns EventAccumulator for a given run.

        Args:
          run: String name of run.

        Returns:
          An EventAccumulator object.

        Raises:
          KeyError: If run does not exist.
        """
    with self._accumulators_mutex:
        return self._accumulators[run]