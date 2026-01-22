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
@retry.RetryOnException(max_retrials=3, should_retry_if=_ShouldRetryMetadataServerConnection)
def _CheckServer(self):
    return gce_read.ReadNoProxy(gce_read.GOOGLE_GCE_METADATA_NUMERIC_PROJECT_URI, properties.VALUES.compute.gce_metadata_check_timeout_sec.GetInt()).isdigit()