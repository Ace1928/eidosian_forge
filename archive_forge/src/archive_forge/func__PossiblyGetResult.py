from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.command_lib.help_search import lookup
from googlecloudsdk.command_lib.help_search import rater
from googlecloudsdk.command_lib.help_search import search_util
from six.moves import zip
def _PossiblyGetResult(self, command):
    """Helper function to determine whether a command contains all terms.

    Returns a copy of the command or command group with modifications to the
    'commands' field and an added 'summary' field if the command matches
    the searcher's search terms.

    Args:
      command: dict, a json representation of a command.

    Returns:
      a modified copy of the command if the command is a result, otherwise None.
    """
    locations = [search_util.LocateTerm(command, term) for term in self.terms]
    if any(locations):
        results = search_util.CommandSearchResults(dict(zip(self.terms, locations)))
        new_command = search_util.ProcessResult(command, results)
        self._rater.AddFoundCommand(new_command, results)
        return new_command