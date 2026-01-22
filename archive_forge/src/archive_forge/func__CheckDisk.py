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
def _CheckDisk(self):
    """Reads cache from disk."""
    gce_cache_path = config.Paths().GCECachePath()
    with self.file_lock:
        try:
            mtime = os.stat(gce_cache_path).st_mtime
            expiration_time = mtime + _GCE_CACHE_MAX_AGE
            gcecache_file_value = files.ReadFileContents(gce_cache_path)
            return (gcecache_file_value == six.text_type(True), expiration_time)
        except (OSError, IOError, files.Error):
            return (None, None)