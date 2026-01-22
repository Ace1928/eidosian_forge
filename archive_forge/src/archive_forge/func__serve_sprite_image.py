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
def _serve_sprite_image(self, request):
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
    embedding_info = self._get_embedding(name, config)
    if not embedding_info or not embedding_info.sprite.image_path:
        return Respond(request, 'No sprite image file found for tensor "%s" in the config file "%s"' % (name, self.config_fpaths[run]), 'text/plain', 400)
    fpath = os.path.expanduser(embedding_info.sprite.image_path)
    fpath = _rel_to_abs_asset_path(fpath, self.config_fpaths[run])
    if not tf.io.gfile.exists(fpath) or tf.io.gfile.isdir(fpath):
        return Respond(request, '"%s" does not exist or is directory' % fpath, 'text/plain', 400)
    f = tf.io.gfile.GFile(fpath, 'rb')
    encoded_image_string = f.read()
    f.close()
    image_type = imghdr.what(None, encoded_image_string)
    mime_type = _IMGHDR_TO_MIMETYPE.get(image_type, _DEFAULT_IMAGE_MIMETYPE)
    return Respond(request, encoded_image_string, mime_type)