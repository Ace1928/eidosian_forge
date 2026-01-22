from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import os
import re
import threading
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import properties_file
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
@classmethod
def _FindFlagValue(cls, args):
    """Parse the given args to find the value of the --configuration flag.

    Args:
      args: [str], The arguments from the command line to parse

    Returns:
      str, The value of the --configuration flag or None if not found.
    """
    flag = '--configuration'
    flag_eq = flag + '='
    successor = None
    config_flag = None
    for arg in reversed(args):
        if arg == flag:
            config_flag = successor
            break
        if arg.startswith(flag_eq):
            _, config_flag = arg.split('=', 1)
            break
        successor = arg
    return config_flag