import imghdr
import urllib.parse
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.image import metadata
@wrappers.Request.application
def _serve_tags(self, request):
    ctx = plugin_util.context(request.environ)
    experiment = plugin_util.experiment_id(request.environ)
    index = self._index_impl(ctx, experiment)
    return http_util.Respond(request, index, 'application/json')