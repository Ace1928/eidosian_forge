import os.path
from tensorboard.compat import tf
Retrieve a particular plugin asset from a logdir.

    Args:
      logdir: A directory that was created by a TensorFlow summary.FileWriter.
      plugin_name: The plugin we want an asset from.
      asset_name: The name of the requested asset.

    Returns:
      string contents of the plugin asset.

    Raises:
      KeyError: if the asset does not exist.
    