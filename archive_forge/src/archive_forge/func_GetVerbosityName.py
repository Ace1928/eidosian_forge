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
def GetVerbosityName(verbosity=None):
    """Gets the name for the current verbosity setting or verbosity if not None.

  Args:
    verbosity: int, Returns the name for this verbosity if not None.

  Returns:
    str, The verbosity name or None if the verbosity is unknown.
  """
    if verbosity is None:
        verbosity = GetVerbosity()
    for name, num in six.iteritems(VALID_VERBOSITY_STRINGS):
        if verbosity == num:
            return name
    return None