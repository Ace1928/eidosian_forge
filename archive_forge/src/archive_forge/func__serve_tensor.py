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
def _serve_tensor(self, request):
    run = request.args.get('run')
    if run is None:
        return Respond(request, 'query parameter "run" is required', 'text/plain', 400)
    name = request.args.get('name')
    if name is None:
        return Respond(request, 'query parameter "name" is required', 'text/plain', 400)
    num_rows = _parse_positive_int_param(request, 'num_rows')
    if num_rows == -1:
        return Respond(request, 'query parameter num_rows must be integer > 0', 'text/plain', 400)
    self._update_configs()
    config = self._configs.get(run)
    if config is None:
        return Respond(request, 'Unknown run: "%s"' % run, 'text/plain', 400)
    tensor = self.tensor_cache.get((run, name))
    if tensor is None:
        embedding = self._get_embedding(name, config)
        if embedding and embedding.tensor_path:
            fpath = _rel_to_abs_asset_path(embedding.tensor_path, self.config_fpaths[run])
            if not tf.io.gfile.exists(fpath):
                return Respond(request, 'Tensor file "%s" does not exist' % fpath, 'text/plain', 400)
            try:
                tensor = _read_tensor_tsv_file(fpath)
            except UnicodeDecodeError:
                tensor = _read_tensor_binary_file(fpath, embedding.tensor_shape)
        else:
            reader = self._get_reader_for_run(run)
            if not reader or not reader.has_tensor(name):
                return Respond(request, 'Tensor "%s" not found in checkpoint dir "%s"' % (name, config.model_checkpoint_path), 'text/plain', 400)
            try:
                tensor = reader.get_tensor(name)
            except tf.errors.InvalidArgumentError as e:
                return Respond(request, str(e), 'text/plain', 400)
        self.tensor_cache.set((run, name), tensor)
    if num_rows:
        tensor = tensor[:num_rows]
    if tensor.dtype != 'float32':
        tensor = tensor.astype(dtype='float32', copy=False)
    data_bytes = tensor.tobytes()
    return Respond(request, data_bytes, 'application/octet-stream')