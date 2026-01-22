from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import sys
from googlecloudsdk.command_lib.interactive import browser
from prompt_toolkit import keys
from prompt_toolkit.key_binding import manager
import six
class _KeyBinding(object):
    """Key binding base info to keep registered bindings and toolbar in sync.

  Attributes:
    key: The keys.Key.* object.
    help_text: The UX help text.
    label: The short word label for the bottom toolbar.
    metavar: Display this value in GetLabel(markdown=True) instead of the real
      value.
    status: The bool => string toggle status map.
    toggle: The bool toggle state.
  """

    def __init__(self, key, help_text=None, label=None, metavar=None, status=None, toggle=True):
        self.key = key
        self.help_text = help_text
        self.label = label
        self.metavar = metavar
        self.status = status
        self.toggle = toggle

    def GetName(self):
        """Returns the binding display name."""
        return re.sub('.*<(.*)>.*', '\\1', six.text_type(self.key)).replace('C-', 'ctrl-')

    def GetLabel(self, markdown=False):
        """Returns the key binding display label containing the name and value."""
        if self.label is None and self.status is None:
            return None
        label = []
        if markdown:
            label.append('*')
        label.append(self.GetName())
        label.append(':')
        if self.label:
            label.append(self.label)
            if self.status:
                label.append(':')
        if markdown:
            label.append('*')
        if self.status:
            if markdown:
                label.append('_')
                label.append(self.metavar or 'STATE')
                label.append('_')
            else:
                label.append(self.status[self.toggle])
        return ''.join(label)

    def GetHelp(self, markdown=False):
        """Returns the key help text."""
        if not self.help_text:
            return None
        key = self.GetName()
        if markdown:
            key = '*{}*'.format(key)
        return self.help_text.format(key=key)

    def SetMode(self, cli):
        """Sets the toggle mode in the cli."""
        del cli

    def Handle(self, event):
        """Handles a bound key event."""
        self.toggle = not self.toggle
        self.SetMode(event.cli)