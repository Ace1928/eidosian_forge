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
def _latest_checkpoints_changed(configs, run_path_pairs):
    """Returns true if the latest checkpoint has changed in any of the runs."""
    for run_name, assets_dir in run_path_pairs:
        if run_name not in configs:
            config = ProjectorConfig()
            config_fpath = os.path.join(assets_dir, metadata.PROJECTOR_FILENAME)
            if tf.io.gfile.exists(config_fpath):
                with tf.io.gfile.GFile(config_fpath, 'r') as f:
                    file_content = f.read()
                text_format.Parse(file_content, config)
        else:
            config = configs[run_name]
        logdir = _assets_dir_to_logdir(assets_dir)
        ckpt_path = _find_latest_checkpoint(logdir)
        if not ckpt_path:
            continue
        if config.model_checkpoint_path != ckpt_path:
            return True
    return False