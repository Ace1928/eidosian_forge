import argparse
import functools
import gzip
import io
import mimetypes
import posixpath
import zipfile
from werkzeug import utils
from werkzeug import wrappers
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.util import grpc_util
from tensorboard.util import tb_logging
from tensorboard import version
@wrappers.Request.application
def _serve_window_properties(self, request):
    """Serve a JSON object containing this TensorBoard's window
        properties."""
    return http_util.Respond(request, {'window_title': self._window_title}, 'application/json')