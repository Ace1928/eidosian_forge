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
@wrappers.Request.application
def _serve_file(self, file_path, request):
    """Returns a resource file."""
    res_path = os.path.join(os.path.dirname(__file__), file_path)
    with open(res_path, 'rb') as read_file:
        mimetype = mimetypes.guess_type(file_path)[0]
        return Respond(request, read_file.read(), content_type=mimetype)