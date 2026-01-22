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
def _serve_environment(self, request):
    """Serve a JSON object describing the TensorBoard parameters."""
    ctx = plugin_util.context(request.environ)
    experiment = plugin_util.experiment_id(request.environ)
    md = self._data_provider.experiment_metadata(ctx, experiment_id=experiment)
    environment = {'version': version.VERSION, 'data_location': md.data_location, 'window_title': self._window_title, 'experiment_name': md.experiment_name, 'experiment_description': md.experiment_description, 'creation_time': md.creation_time}
    if self._include_debug_info:
        environment['debug'] = {'data_provider': str(self._data_provider), 'flags': self._render_flags()}
    return http_util.Respond(request, environment, 'application/json')