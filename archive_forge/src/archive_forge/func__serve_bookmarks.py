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
def _serve_bookmarks(self, request):
    run = request.args.get('run')
    if not run:
        return Respond(request, 'query parameter "run" is required', 'text/plain', 400)
    name = request.args.get('name')
    if name is None:
        return Respond(request, 'query parameter "name" is required', 'text/plain', 400)
    self._update_configs()
    config = self._configs.get(run)
    if config is None:
        return Respond(request, 'Unknown run: "%s"' % run, 'text/plain', 400)
    fpath = self._get_bookmarks_file_for_tensor(name, config)
    if not fpath:
        return Respond(request, 'No bookmarks file found for tensor "%s" in the config file "%s"' % (name, self.config_fpaths[run]), 'text/plain', 400)
    fpath = _rel_to_abs_asset_path(fpath, self.config_fpaths[run])
    if not tf.io.gfile.exists(fpath) or tf.io.gfile.isdir(fpath):
        return Respond(request, '"%s" not found, or is not a file' % fpath, 'text/plain', 400)
    bookmarks_json = None
    with tf.io.gfile.GFile(fpath, 'rb') as f:
        bookmarks_json = f.read()
    return Respond(request, bookmarks_json, 'application/json')