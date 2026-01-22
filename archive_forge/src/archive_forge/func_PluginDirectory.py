import os.path
from tensorboard.compat import tf
def PluginDirectory(logdir, plugin_name):
    """Returns the plugin directory for plugin_name."""
    return os.path.join(logdir, _PLUGINS_DIR, plugin_name)