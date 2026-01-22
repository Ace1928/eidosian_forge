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
def PushFromArgs(self, args):
    """Parse the args and add the value that was found to the top of the stack.

    Args:
      args: [str], The command line args for this invocation.
    """
    self.Push(_FlagOverrideStack._FindFlagValue(args))