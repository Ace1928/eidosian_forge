from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import contextlib
import os
import random
import re
import socket
import subprocess
import tempfile
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import portpicker
import six
def PrefixOutput(process, prefix):
    """Prepends the given prefix to each line of the given process's output.

  Args:
    process: process, The handle to the process whose output should be prefixed
    prefix: str, The prefix to be prepended to the process's output.
  """
    output_line = process.stdout.readline()
    while output_line:
        log.status.Print('[{0}] {1}'.format(prefix, encoding.Decode(output_line.rstrip())))
        log.status.flush()
        output_line = process.stdout.readline()