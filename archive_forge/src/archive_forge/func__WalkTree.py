from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.command_lib.help_search import lookup
from googlecloudsdk.command_lib.help_search import rater
from googlecloudsdk.command_lib.help_search import search_util
from six.moves import zip
def _WalkTree(self, current_parent, found_commands):
    """Recursively walks command tree, checking for matches.

    If a command matches, it is postprocessed and added to found_commands.

    Args:
      current_parent: dict, a json representation of a CLI command.
      found_commands: [dict], a list of matching commands.

    Returns:
      [dict], a list of commands that have matched so far.
    """
    result = self._PossiblyGetResult(current_parent)
    if result:
        found_commands.append(result)
    for child_command in current_parent.get(lookup.COMMANDS, {}).values():
        found_commands = self._WalkTree(child_command, found_commands)
    return found_commands