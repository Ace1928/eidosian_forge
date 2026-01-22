from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import json
import os
import pickle
import platform
import socket
import subprocess
import sys
import tempfile
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
import six
import six.moves.urllib.error
import six.moves.urllib.parse
import six.moves.urllib.request
@staticmethod
def ResetCollectorInstance(disable_cache=None):
    """Reset the singleton _MetricsCollector and reinitialize it.

    This should only be used for tests, where we want to collect some metrics
    but not others, and we have to reinitialize the collector with a different
    Google Analytics tracking id.

    Args:
      disable_cache: Metrics collector keeps an internal cache of the disabled
          state of metrics. This controls the value to reinitialize the cache.
          None means we will refresh the cache with the default values.
          True/False forces a specific value.
    """
    _MetricsCollector._disabled_cache = disable_cache
    if _MetricsCollector._IsDisabled():
        _MetricsCollector._instance = None
    else:
        _MetricsCollector._instance = _MetricsCollector()