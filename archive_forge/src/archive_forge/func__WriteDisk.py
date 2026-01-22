from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import socket
import threading
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import gce_read
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import retry
import six
from six.moves import http_client
from six.moves import urllib_error
def _WriteDisk(self, on_gce):
    """Updates cache on disk."""
    gce_cache_path = config.Paths().GCECachePath()
    with self.file_lock:
        try:
            files.WriteFileContents(gce_cache_path, six.text_type(on_gce), private=True)
        except (OSError, IOError, files.Error):
            pass