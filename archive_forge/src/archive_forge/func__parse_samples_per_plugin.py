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
def _parse_samples_per_plugin(value):
    """Parses `value` as a string-to-int dict in the form `foo=12,bar=34`."""
    result = {}
    for token in value.split(','):
        if token:
            k, v = token.strip().split('=')
            result[k] = int(v)
    return result