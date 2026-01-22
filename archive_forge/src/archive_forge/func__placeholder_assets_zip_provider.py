import base64
import collections
import hashlib
import io
import json
import re
import textwrap
import time
from urllib import parse as urlparse
import zipfile
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import auth_context_middleware
from tensorboard.backend import client_feature_flags
from tensorboard.backend import empty_path_redirect
from tensorboard.backend import experiment_id
from tensorboard.backend import experimental_plugin
from tensorboard.backend import http_util
from tensorboard.backend import path_prefix
from tensorboard.backend import security_validator
from tensorboard.plugins import base_plugin
from tensorboard.plugins.core import core_plugin
from tensorboard.util import tb_logging
def _placeholder_assets_zip_provider():
    """Defines a default assets zip provider containing a dummy index.html."""
    memfile = io.BytesIO()
    with zipfile.ZipFile(memfile, mode='w') as zf:
        zf.writestr('index.html', 'TensorBoard placeholder index.html')
    return memfile