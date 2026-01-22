from google.protobuf import message
import requests
from absl import logging
from tensorboard import version
from tensorboard.plugins.scalar import metadata as scalars_metadata
from tensorboard.uploader.proto import server_info_pb2
def allowed_plugins(server_info):
    """Determines which plugins may upload data.

    This pulls from the `plugin_control` on the `server_info` when that
    submessage is set, else falls back to a default.

    Args:
      server_info: A `server_info_pb2.ServerInfoResponse` message.

    Returns:
      A `frozenset` of plugin names.
    """
    if server_info.HasField('plugin_control'):
        return frozenset(server_info.plugin_control.allowed_plugins)
    else:
        return frozenset((scalars_metadata.PLUGIN_NAME,))