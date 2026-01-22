from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.core.console import console_io
import six
def _IsLocal(uri):
    """Checks if a given uri represent a local file."""
    drive, _ = os.path.splitdrive(uri)
    parsed_uri = six.moves.urllib.parse.urlsplit(uri, allow_fragments=False)
    return drive or not parsed_uri.scheme