from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import datetime
import os.path
import signal
import subprocess
import sys
import threading
from googlecloudsdk.command_lib.code import json_stream
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files as file_utils
import six
class PrintUrlThreadContext(object):
    """Context manager that starts a thread that prints outs local urls.

  When entering the context, start a thread that watches the skaffold events
  stream api, find the portForward events, and prints out the local urls
  for a service. This will continue until the context is exited.
  """

    def __init__(self, service_name, events_port):
        """Initialize PrintUrlThreadContext.

    Args:
      service_name: Name of the service.
      events_port: Port number of the skaffold events stream api.
    """
        self._stop = threading.Event()
        self._thread = threading.Thread(target=_PrintUrl, args=(service_name, events_port, self._stop))

    def __enter__(self):
        self._thread.start()

    def __exit__(self, *args):
        self._stop.set()