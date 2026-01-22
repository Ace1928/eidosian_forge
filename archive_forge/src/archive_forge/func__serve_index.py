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
def _serve_index(self, index_asset_bytes, request):
    """Serves index.html content.

        Note that we opt out of gzipping index.html to write preamble before the
        resource content. This inflates the resource size from 2x kiB to 1xx
        kiB, but we require an ability to flush preamble with the HTML content.
        """
    relpath = posixpath.relpath(self._path_prefix, request.script_root) if self._path_prefix else '.'
    meta_header = '<!doctype html><meta name="tb-relative-root" content="%s/">' % relpath
    content = meta_header.encode('utf-8') + index_asset_bytes
    return http_util.Respond(request, content, 'text/html', content_encoding='identity')