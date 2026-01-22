from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import os
import signal
import sys
import threading
import time
import enum
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import multiline
from googlecloudsdk.core.console.style import parser
import six
def ProgressTracker(message=None, autotick=True, detail_message_callback=None, done_message_callback=None, tick_delay=0.2, interruptable=True, screen_reader=False, aborted_message=console_io.OperationCancelledError.DEFAULT_MESSAGE, no_spacing=False):
    """A context manager for telling the user about long-running progress.

  Args:
    message: str, The message to show next to the spinner.
    autotick: bool, True to have the spinner tick on its own. Otherwise, you
      need to call Tick() explicitly to move the spinner.
    detail_message_callback: func, A no argument function that will be called
      and the result appended to message each time it needs to be printed.
    done_message_callback: func, A no argument function whose result will be
      appended to message if the progress tracker successfully exits.
    tick_delay: float, The amount of time to wait between ticks, in second.
    interruptable: boolean, True if the user can ctrl-c the operation. If so,
      it will stop and will report as aborted. If False, a message will be
      displayed saying that it cannot be cancelled.
    screen_reader: boolean, override for screen reader accessibility property
      toggle.
    aborted_message: str, A custom message to put in the exception when it is
      cancelled by the user.
    no_spacing: boolean, Removes ellipses and other spacing between text.

  Returns:
    The progress tracker.
  """
    style = properties.VALUES.core.interactive_ux_style.Get()
    if style == properties.VALUES.core.InteractiveUXStyles.OFF.name:
        return NoOpProgressTracker(interruptable, aborted_message)
    elif style == properties.VALUES.core.InteractiveUXStyles.TESTING.name:
        return _StubProgressTracker(message, interruptable, aborted_message)
    else:
        is_tty = console_io.IsInteractive(error=True)
        tracker_cls = _NormalProgressTracker if is_tty else _NonInteractiveProgressTracker
        screen_reader = screen_reader or properties.VALUES.accessibility.screen_reader.GetBool()
        spinner_override_message = None
        if screen_reader:
            tick_delay = 1
            spinner_override_message = 'working'
        return tracker_cls(message, autotick, detail_message_callback, done_message_callback, tick_delay, interruptable, aborted_message, spinner_override_message, no_spacing)