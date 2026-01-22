from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import contextlib
import enum
import getpass
import io
import json
import os
import re
import subprocess
import sys
import textwrap
import threading
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_pager
from googlecloudsdk.core.console import prompt_completer
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
from six.moves import input  # pylint: disable=redefined-builtin
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import range  # pylint: disable=redefined-builtin
class _TextPercentageProgressBar(object):
    """A progress bar that outputs nothing at all."""

    def __init__(self, label, stream, percentage_display_increments=5.0):
        """Creates a progress bar for the given action.

    Args:
      label: str, The action that is being performed.
      stream: The output stream to write to, stderr by default.
      percentage_display_increments: Minimum change in percetnage to display new
        progress
    """
        self._label = label
        self._stream = stream
        self._last_percentage = 0
        self._percentage_display_increments = percentage_display_increments / 100.0

    def Start(self):
        self._Write(self._label)

    def SetProgress(self, progress_factor):
        progress_factor = min(progress_factor, 1.0)
        should_update_progress = progress_factor > self._last_percentage + self._percentage_display_increments
        if should_update_progress or progress_factor == 1.0:
            self._Write('{0:.0f}%'.format(progress_factor * 100.0))
            self._last_percentage = progress_factor

    def Finish(self):
        """Mark the progress as done."""
        self.SetProgress(1)

    def _Write(self, msg):
        self._stream.write(msg + '\n')

    def __enter__(self):
        self.Start()
        return self

    def __exit__(self, *args):
        self.Finish()