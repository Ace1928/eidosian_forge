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
def _ShouldRetryMetadataServerConnection(exc_type, exc_value, exc_traceback, state):
    """Decides if we need to retry the metadata server connection."""
    del exc_type, exc_traceback, state
    if not isinstance(exc_value, _POSSIBLE_ERRORS_GCE_METADATA_CONNECTION):
        return False
    if isinstance(exc_value, urllib_error.URLError) and _DOMAIN_NAME_RESOLVE_ERROR_MSG in six.text_type(exc_value):
        return False
    return True