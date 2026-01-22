from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import walker
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_projector
class CoverageCommandNode(dict):
    """Command/group info.

  Attributes:
    commands: {str:_Command}, The subcommands in a command group.
    flags: [str], Command flag list. Global flags, available to all commands,
      are in the root command flags list.
  """

    def __init__(self, command, parent):
        super(CoverageCommandNode, self).__init__()
        self._parent = parent
        if parent is not None:
            name = command.name.replace('_', '-')
            parent[name] = self
        args = command.ai
        for arg in args.flag_args:
            for name in arg.option_strings:
                if arg.is_hidden:
                    continue
                if not name.startswith('--'):
                    continue
                if self.IsAncestorFlag(name):
                    continue
                self[name] = arg.require_coverage_in_tests

    def IsAncestorFlag(self, flag):
        """Determines if flag is provided by an ancestor command.

    NOTE: This function is used to allow for global flags to be added in at the
          top level but not in subgroups/commands
    Args:
      flag: str, The flag name (no leading '-').

    Returns:
      bool, True if flag provided by an ancestor command, false if not.
    """
        command = self._parent
        while command:
            if flag in command:
                return True
            command = command._parent
        return False