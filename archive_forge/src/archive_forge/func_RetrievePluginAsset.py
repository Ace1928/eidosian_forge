import os
import threading
from typing import Optional
from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.util import tb_logging
def RetrievePluginAsset(self, run, plugin_name, asset_name):
    """Return the contents for a specific plugin asset from a run.

        Args:
          run: The string name of the run.
          plugin_name: The string name of a plugin.
          asset_name: The string name of an asset.

        Returns:
          The string contents of the plugin asset.

        Raises:
          KeyError: If the asset is not available.
        """
    accumulator = self.GetAccumulator(run)
    return accumulator.RetrievePluginAsset(plugin_name, asset_name)