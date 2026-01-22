import imghdr
import urllib.parse
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.image import metadata
def _data_provider_query(self, blob_reference):
    return urllib.parse.urlencode({'blob_key': blob_reference.blob_key})