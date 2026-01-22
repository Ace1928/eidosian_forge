import collections
import functools
import imghdr
import mimetypes
import os
import threading
import numpy as np
from werkzeug import wrappers
from google.protobuf import json_format
from google.protobuf import text_format
from tensorboard import context
from tensorboard.backend.event_processing import plugin_asset_util
from tensorboard.backend.http_util import Respond
from tensorboard.compat import tf
from tensorboard.plugins import base_plugin
from tensorboard.plugins.projector import metadata
from tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig
from tensorboard.util import tb_logging
def _determine_is_active(self):
    """Determines whether the plugin is active.

        This method is run in a separate thread so that the plugin can
        offer an immediate response to whether it is active and
        determine whether it should be active in a separate thread.
        """
    self._update_configs()
    if self._configs:
        self._is_active = True
    self._thread_for_determining_is_active = None