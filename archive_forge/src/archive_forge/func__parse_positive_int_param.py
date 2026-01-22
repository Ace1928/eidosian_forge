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
def _parse_positive_int_param(request, param_name):
    """Parses and asserts a positive (>0) integer query parameter.

    Args:
      request: The Werkzeug Request object
      param_name: Name of the parameter.

    Returns:
      Param, or None, or -1 if parameter is not a positive integer.
    """
    param = request.args.get(param_name)
    if not param:
        return None
    try:
        param = int(param)
        if param <= 0:
            raise ValueError()
        return param
    except ValueError:
        return -1