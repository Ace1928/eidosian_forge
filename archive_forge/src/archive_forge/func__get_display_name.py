import json
import tensorflow as tf
from tensorboard.plugins.mesh import metadata
from tensorboard.plugins.mesh import plugin_data_pb2
from tensorboard.plugins.mesh import summary_v2
def _get_display_name(name, display_name):
    """Returns display_name from display_name and name."""
    if display_name is None:
        return name
    return display_name