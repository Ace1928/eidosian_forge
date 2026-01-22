import os
import threading
from typing import Optional
from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.util import tb_logging
def GetSourceWriter(self, run) -> Optional[str]:
    """Returns the source writer name from the first event of the given run.

        Assuming each run has only one source writer.

        Args:
          run: A string name of the run from which the event source information
            is retrieved.

        Returns:
          Name of the writer that wrote the events in the run.
        """
    accumulator = self.GetAccumulator(run)
    return accumulator.GetSourceWriter()