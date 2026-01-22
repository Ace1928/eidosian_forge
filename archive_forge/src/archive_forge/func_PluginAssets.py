import os
import threading
from typing import Optional
from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.util import tb_logging
def PluginAssets(self, plugin_name):
    """Get index of runs and assets for a given plugin.

        Args:
          plugin_name: Name of the plugin we are checking for.

        Returns:
          A dictionary that maps from run_name to a list of plugin
            assets for that run.
        """
    with self._accumulators_mutex:
        items = list(self._accumulators.items())
    return {run: accum.PluginAssets(plugin_name) for run, accum in items}