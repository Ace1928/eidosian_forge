import imghdr
import urllib.parse
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.image import metadata
def _filter_by_sample(self, tensor_events, sample):
    return [tensor_event for tensor_event in tensor_events if len(tensor_event.tensor_proto.string_val) - 2 > sample]