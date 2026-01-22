from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
import os
import subprocess
import sys
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.credentials import store
from googlecloudsdk.core.docker import client_lib
from googlecloudsdk.core.docker import constants
from googlecloudsdk.core.util import files
import six
def _IsExpectedErrorLine(line):
    """Returns whether or not the given line was expected from the Docker client.

  Args:
    line: The line received in stderr from Docker
  Returns:
    True if the line was expected, False otherwise.
  """
    expected_line_substrs = ['--email', 'login credentials saved in', 'WARNING! Using --password via the CLI is insecure. Use --password-stdin.']
    for expected_line_substr in expected_line_substrs:
        if expected_line_substr in line:
            return True
    return False