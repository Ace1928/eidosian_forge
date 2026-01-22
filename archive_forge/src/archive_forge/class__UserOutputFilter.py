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
class _UserOutputFilter(object):
    """A filter to turn on and off user output.

  This filter is used by the ConsoleWriter to determine if output messages
  should be printed or not.
  """

    def __init__(self, enabled):
        """Creates the filter.

    Args:
      enabled: bool, True to enable output, false to suppress.
    """
        self.enabled = enabled