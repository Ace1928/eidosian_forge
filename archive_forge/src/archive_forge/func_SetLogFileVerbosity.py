from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from collections import OrderedDict
import contextlib
import copy
import datetime
import json
import logging
import os
import sys
import time
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console.style import parser as style_parser
from googlecloudsdk.core.console.style import text
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def SetLogFileVerbosity(verbosity):
    """Sets the log file verbosity.

  Args:
    verbosity: int, A verbosity constant from the logging module that
      determines what level of logs will be written to the log file. If None,
      the default will be used.

  Returns:
    int, The current verbosity.
  """
    return _GetEffectiveVerbosity(_log_manager.file_only_logger.setLevel(verbosity))