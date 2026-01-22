import threading
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.debugger_v2 import debug_data_provider
from tensorboard.backend import http_util
Serves the content of stack frames.

        The source frames being requested are referred to be UUIDs for each of
        them, separated by commas.

        Args:
          request: HTTP request.

        Returns:
          Response to the request.
        