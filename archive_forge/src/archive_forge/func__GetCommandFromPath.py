from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.calliope import markdown
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import properties
import six
def _GetCommandFromPath(self, command_path):
    """Returns the command node for command_path."""
    path = self._tree[cli_tree.LOOKUP_PATH]
    if path:
        if command_path[:1] != path:
            return None
        command_path = command_path[1:]
    command = self._tree
    for name in command_path:
        commands = command[cli_tree.LOOKUP_COMMANDS]
        if name not in commands:
            return None
        command = commands[name]
    return command