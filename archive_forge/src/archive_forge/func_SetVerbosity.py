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
def SetVerbosity(self, verbosity):
    """Sets the active verbosity for the logger.

    Args:
      verbosity: int, A verbosity constant from the logging module that
        determines what level of logs will show in the console. If None, the
        value from properties or the default will be used.

    Returns:
      int, The current verbosity.
    """
    if verbosity is None:
        verbosity_string = properties.VALUES.core.verbosity.Get()
        if verbosity_string is not None:
            verbosity = VALID_VERBOSITY_STRINGS.get(verbosity_string.lower())
    if verbosity is None:
        verbosity = DEFAULT_VERBOSITY
    if self.verbosity == verbosity:
        return self.verbosity
    self.stderr_handler.setLevel(verbosity)
    old_verbosity = self.verbosity
    self.verbosity = verbosity
    return old_verbosity