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
def _append_plugin_asset_directories(self, run_path_pairs):
    extra = []
    plugin_assets_name = metadata.PLUGIN_ASSETS_NAME
    for run, logdir in run_path_pairs:
        assets = plugin_asset_util.ListAssets(logdir, plugin_assets_name)
        if metadata.PROJECTOR_FILENAME not in assets:
            continue
        assets_dir = os.path.join(self._run_paths[run], metadata.PLUGINS_DIR, plugin_assets_name)
        assets_path_pair = (run, os.path.abspath(assets_dir))
        extra.append(assets_path_pair)
    run_path_pairs.extend(extra)