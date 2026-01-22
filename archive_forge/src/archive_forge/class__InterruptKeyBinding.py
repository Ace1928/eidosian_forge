from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import sys
from googlecloudsdk.command_lib.interactive import browser
from prompt_toolkit import keys
from prompt_toolkit.key_binding import manager
import six
class _InterruptKeyBinding(_KeyBinding):
    """The interrupt (ctrl-c) key binding.

  Catches control-C and clears the prompt input buffer and completer.
  """

    def __init__(self, key):
        super(_InterruptKeyBinding, self).__init__(key=key)

    def Handle(self, event):
        event.cli.current_buffer.reset()
        event.cli.completer.reset()