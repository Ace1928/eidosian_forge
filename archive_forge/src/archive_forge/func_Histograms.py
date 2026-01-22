import os
import threading
from typing import Optional
from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.util import tb_logging
def Histograms(self, run, tag):
    """Retrieve the histogram events associated with a run and tag.

        Args:
          run: A string name of the run for which values are retrieved.
          tag: A string name of the tag for which values are retrieved.

        Raises:
          KeyError: If the run is not found, or the tag is not available for
            the given run.

        Returns:
          An array of `event_accumulator.HistogramEvents`.
        """
    accumulator = self.GetAccumulator(run)
    return accumulator.Histograms(tag)