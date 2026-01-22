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
class _NormalProgressBar(object):
    """A simple progress bar for tracking completion of an action.

  This progress bar works without having to use any control characters.  It
  prints the action that is being done, and then fills a progress bar below it.
  You should not print anything else on the output stream during this time as it
  will cause the progress bar to break on lines.

  Progress bars can be stacked into a group. first=True marks the first bar in
  the group and last=True marks the last bar in the group. The default assumes
  a singleton bar with first=True and last=True.

  This class can also be used in a context manager.
  """

    def __init__(self, label, stream, total_ticks, first, last):
        """Creates a progress bar for the given action.

    Args:
      label: str, The action that is being performed.
      stream: The output stream to write to, stderr by default.
      total_ticks: int, The number of ticks wide to make the progress bar.
      first: bool, True if this is the first bar in a stacked group.
      last: bool, True if this is the last bar in a stacked group.
    """
        self._raw_label = label
        self._stream = stream
        self._ticks_written = 0
        self._total_ticks = total_ticks
        self._first = first
        self._last = last
        attr = console_attr.ConsoleAttr()
        self._box = attr.GetBoxLineCharacters()
        self._redraw = self._box.d_dr != self._box.d_vr or self._box.d_dl != self._box.d_vl
        if self._redraw and (not IsInteractive(error=True)):
            self._first = True
            self._last = True
        max_label_width = self._total_ticks - 4
        if len(label) > max_label_width:
            label = label[:max_label_width - 3] + '...'
        elif len(label) < max_label_width:
            diff = max_label_width - len(label)
            label += ' ' * diff
        left = self._box.d_vr + self._box.d_h
        right = self._box.d_h + self._box.d_vl
        self._label = '{left} {label} {right}'.format(left=left, label=label, right=right)

    def Start(self):
        """Starts the progress bar by writing the top rule and label."""
        if self._first or self._redraw:
            left = self._box.d_dr if self._first else self._box.d_vr
            right = self._box.d_dl if self._first else self._box.d_vl
            rule = '{left}{middle}{right}\n'.format(left=left, middle=self._box.d_h * self._total_ticks, right=right)
            self._Write(rule)
        self._Write(self._label + '\n')
        self._Write(self._box.d_ur)
        self._ticks_written = 0

    def SetProgress(self, progress_factor):
        """Sets the current progress of the task.

    This method has no effect if the progress bar has already progressed past
    the progress you call it with (since the progress bar cannot back up).

    Args:
      progress_factor: float, The current progress as a float between 0 and 1.
    """
        expected_ticks = int(self._total_ticks * progress_factor)
        new_ticks = expected_ticks - self._ticks_written
        new_ticks = min(new_ticks, self._total_ticks - self._ticks_written)
        if new_ticks > 0:
            self._Write(self._box.d_h * new_ticks)
            self._ticks_written += new_ticks
            if expected_ticks == self._total_ticks:
                end = '\n' if self._last or not self._redraw else '\r'
                self._Write(self._box.d_ul + end)
            self._stream.flush()

    def Finish(self):
        """Mark the progress as done."""
        self.SetProgress(1)

    def _Write(self, msg):
        self._stream.write(msg)

    def __enter__(self):
        self.Start()
        return self

    def __exit__(self, *args):
        self.Finish()