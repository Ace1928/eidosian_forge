import threading
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.debugger_v2 import debug_data_provider
from tensorboard.backend import http_util
def _error_response(request, error_message):
    return http_util.Respond(request, {'error': error_message}, 'application/json', code=400)