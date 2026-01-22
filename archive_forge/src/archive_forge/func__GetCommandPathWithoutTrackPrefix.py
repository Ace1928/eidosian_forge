from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.command_lib.help_search import lookup
from googlecloudsdk.command_lib.help_search import rater
from googlecloudsdk.command_lib.help_search import search_util
from six.moves import zip
def _GetCommandPathWithoutTrackPrefix(command):
    """Helper to get the path of a command without a track prefix.

  Args:
    command: dict, json representation of a command.

  Returns:
    a ' '-separated string representation of a command path without any
      track prefixes.
  """
    return ' '.join([segment for segment in command[lookup.PATH] if segment not in [lookup.ALPHA_PATH, lookup.BETA_PATH]])