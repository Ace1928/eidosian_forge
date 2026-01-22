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
def CheckServerRefreshAllCaches(self):
    on_gce = self._CheckServerWithRetry()
    self._WriteDisk(on_gce)
    self._WriteMemory(on_gce, time.time() + _GCE_CACHE_MAX_AGE)
    return on_gce