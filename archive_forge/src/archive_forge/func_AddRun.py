import os
import threading
from typing import Optional
from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.util import tb_logging
def AddRun(self, path, name=None):
    """Add a run to the multiplexer.

        If the name is not specified, it is the same as the path.

        If a run by that name exists, and we are already watching the right path,
          do nothing. If we are watching a different path, replace the event
          accumulator.

        If `Reload` has been called, it will `Reload` the newly created
        accumulators.

        Args:
          path: Path to the event files (or event directory) for given run.
          name: Name of the run to add. If not provided, is set to path.

        Returns:
          The `EventMultiplexer`.
        """
    name = name or path
    accumulator = None
    with self._accumulators_mutex:
        if name not in self._accumulators or self._paths[name] != path:
            if name in self._paths and self._paths[name] != path:
                logger.warning('Conflict for name %s: old path %s, new path %s', name, self._paths[name], path)
            logger.info('Constructing EventAccumulator for %s', path)
            accumulator = event_accumulator.EventAccumulator(path, size_guidance=self._size_guidance, purge_orphaned_data=self.purge_orphaned_data)
            self._accumulators[name] = accumulator
            self._paths[name] = path
    if accumulator:
        if self._reload_called:
            accumulator.Reload()
    return self