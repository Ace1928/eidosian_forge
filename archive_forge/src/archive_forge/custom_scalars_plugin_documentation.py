import re
from google.protobuf import json_format
from werkzeug import wrappers
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.compat import tf
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorboard.plugins.custom_scalar import metadata
from tensorboard.plugins.scalar import metadata as scalars_metadata
from tensorboard.plugins.scalar import scalars_plugin
Fetches the custom layout specified by the config file in the logdir.

        If more than 1 run contains a layout, this method merges the layouts by
        merging charts within individual categories. If 2 categories with the same
        name are found, the charts within are merged. The merging is based on the
        order of the runs to which the layouts are written.

        The response is a JSON object mirroring properties of the Layout proto if a
        layout for any run is found.

        The response is an empty object if no layout could be found.
        