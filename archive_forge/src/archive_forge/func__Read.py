from source directory to docker image. They are stored as templated .yaml files
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import enum
import os
import re
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import config as cloudbuild_config
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
import six
import six.moves.urllib.error
import six.moves.urllib.parse
import six.moves.urllib.request
@contextlib.contextmanager
def _Read(uri):
    """Read a file/object (local file:// or gs:// Cloud Storage path).

  >>> with _Read('gs://builder/object.txt') as f:
  ...   assert f.read() == 'foo'
  >>> with _Read('file:///path/to/object.txt') as f:
  ...   assert f.read() == 'bar'

  Args:
    uri: str, the path to the file/object to read. Must begin with 'file://' or
      'gs://'

  Yields:
    a file-like context manager.

  Raises:
    FileReadError: If opening or reading the file failed.
    InvalidRuntimeBuilderPath: If the path is invalid (doesn't begin with an
        appropriate prefix).
  """
    try:
        if uri.startswith('file://'):
            with contextlib.closing(six.moves.urllib.request.urlopen(uri)) as req:
                yield req
        elif uri.startswith('gs://'):
            storage_client = storage_api.StorageClient()
            object_ = storage_util.ObjectReference.FromUrl(uri)
            with contextlib.closing(storage_client.ReadObject(object_)) as f:
                yield f
        else:
            raise InvalidRuntimeBuilderURI(uri)
    except (six.moves.urllib.error.HTTPError, six.moves.urllib.error.URLError, calliope_exceptions.BadFileException) as e:
        log.debug('', exc_info=True)
        raise FileReadError(six.text_type(e))