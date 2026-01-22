import os
import threading
from typing import Optional
from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.util import tb_logging
def AddRunsFromDirectory(self, path, name=None):
    """Load runs from a directory; recursively walks subdirectories.

        If path doesn't exist, no-op. This ensures that it is safe to call
          `AddRunsFromDirectory` multiple times, even before the directory is made.

        If path is a directory, load event files in the directory (if any exist) and
          recursively call AddRunsFromDirectory on any subdirectories. This mean you
          can call AddRunsFromDirectory at the root of a tree of event logs and
          TensorBoard will load them all.

        If the `EventMultiplexer` is already loaded this will cause
        the newly created accumulators to `Reload()`.
        Args:
          path: A string path to a directory to load runs from.
          name: Optionally, what name to apply to the runs. If name is provided
            and the directory contains run subdirectories, the name of each subrun
            is the concatenation of the parent name and the subdirectory name. If
            name is provided and the directory contains event files, then a run
            is added called "name" and with the events from the path.

        Raises:
          ValueError: If the path exists and isn't a directory.

        Returns:
          The `EventMultiplexer`.
        """
    logger.info('Starting AddRunsFromDirectory: %s', path)
    for subdir in io_wrapper.GetLogdirSubdirectories(path):
        logger.info('Adding events from directory %s', subdir)
        rpath = os.path.relpath(subdir, path)
        subname = os.path.join(name, rpath) if name else rpath
        self.AddRun(subdir, name=subname)
    logger.info('Done with AddRunsFromDirectory: %s', path)
    return self