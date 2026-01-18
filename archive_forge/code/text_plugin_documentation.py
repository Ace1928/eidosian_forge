import textwrap
import numpy as np
from werkzeug import wrappers
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.text import metadata
Instantiates TextPlugin via TensorBoard core.

        Args:
          context: A base_plugin.TBContext instance.
        