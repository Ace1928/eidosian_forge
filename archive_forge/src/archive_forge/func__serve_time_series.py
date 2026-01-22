import collections
import imghdr
import json
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.histogram import metadata as histogram_metadata
from tensorboard.plugins.image import metadata as image_metadata
from tensorboard.plugins.metrics import metadata
from tensorboard.plugins.scalar import metadata as scalar_metadata
@wrappers.Request.application
def _serve_time_series(self, request):
    ctx = plugin_util.context(request.environ)
    experiment = plugin_util.experiment_id(request.environ)
    if request.method == 'POST':
        series_requests_string = request.form.get('requests')
    else:
        series_requests_string = request.args.get('requests')
    if not series_requests_string:
        raise errors.InvalidArgumentError("Missing 'requests' field")
    try:
        series_requests = json.loads(series_requests_string)
    except ValueError:
        raise errors.InvalidArgumentError("Unable to parse 'requests' as JSON")
    response = self._time_series_impl(ctx, experiment, series_requests)
    return http_util.Respond(request, response, 'application/json')