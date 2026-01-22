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
def _OpenEventStreamRetry(events_port, stop_event, retry_interval=datetime.timedelta(seconds=1)):
    """Open a connection to the skaffold events api output.

  This function retries opening the connection until opening is succesful or
  stop_event is set.

  Args:
    events_port: Port of the events api.
    stop_event: A threading.Event object.
    retry_interval: Interval for which to sleep between tries.

  Returns:
    urlopen response.
  Raises:
    StopThreadError: The stop_event was set before a connection was established.
  """
    while not stop_event.is_set():
        try:
            return OpenEventsStream(events_port)
        except six.moves.urllib.error.URLError:
            stop_event.wait(retry_interval.total_seconds())
    raise StopThreadError()