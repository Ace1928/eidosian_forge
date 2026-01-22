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
def GetServiceLocalPort(response, service_name):
    """Get the local port for a service.

  This function yields the new local port every time a new port forwarding
  connection is created.

  Args:
    response: urlopen response.
    service_name: Name of the service.

  Yields:
    Local port number.
  """
    for event in ReadEventStream(response):
        if _IsPortEventForService(event, service_name):
            yield event['portEvent']['localPort']