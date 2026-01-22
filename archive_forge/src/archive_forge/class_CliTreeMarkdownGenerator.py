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
class CliTreeMarkdownGenerator(markdown.MarkdownGenerator):
    """cli_tree command help markdown document generator.

  Attributes:
    _capsule: The help text capsule.
    _command: The tree node for command.
    _command_path: The command path list.
    _tree: The (sub)tree root.
    _sections: The help text sections indexed by SECTION name.
    _subcommands: The dict of subcommand help indexed by subcommand name.
    _subgroups: The dict of subgroup help indexed by subcommand name.
  """

    def __init__(self, command, tree):
        """Constructor.

    Args:
      command: The command node in the root tree.
      tree: The (sub)tree root.
    """
        self._tree = tree
        self._command = command
        self._command_path = command[cli_tree.LOOKUP_PATH]
        super(CliTreeMarkdownGenerator, self).__init__(self._command_path, _GetReleaseTrackFromId(self._command[cli_tree.LOOKUP_RELEASE]), self._command.get(cli_tree.LOOKUP_IS_HIDDEN, self._command.get('hidden', False)))
        self._capsule = self._command[cli_tree.LOOKUP_CAPSULE]
        self._sections = self._command[cli_tree.LOOKUP_SECTIONS]
        self._subcommands = self.GetSubCommandHelp()
        self._subgroups = self.GetSubGroupHelp()

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

    def IsValidSubPath(self, command_path):
        """Returns True if the given command path after the top is valid."""
        return self._GetCommandFromPath([cli_tree.DEFAULT_CLI_NAME] + command_path) is not None

    def GetArguments(self):
        """Returns the command arguments."""
        command = self._GetCommandFromPath(self._command_path)
        try:
            return [Argument(a) for a in command[cli_tree.LOOKUP_CONSTRAINTS][cli_tree.LOOKUP_ARGUMENTS]]
        except (KeyError, TypeError):
            return []

    def GetArgDetails(self, arg, depth=None):
        """Returns the help text with auto-generated details for arg.

    The help text was already generated on the cli_tree generation side.

    Args:
      arg: The arg to auto-generate help text for.
      depth: The indentation depth at which the details should be printed.
        Added here only to maintain consistency with superclass during testing.

    Returns:
      The help text with auto-generated details for arg.
    """
        return arg.help

    def _GetSubHelp(self, is_group=False):
        """Returns the help dict indexed by command for sub commands or groups."""
        return {name: usage_text.HelpInfo(help_text=subcommand[cli_tree.LOOKUP_CAPSULE], is_hidden=subcommand.get(cli_tree.LOOKUP_IS_HIDDEN, subcommand.get('hidden', False)), release_track=_GetReleaseTrackFromId(subcommand[cli_tree.LOOKUP_RELEASE])) for name, subcommand in six.iteritems(self._command[cli_tree.LOOKUP_COMMANDS]) if subcommand[cli_tree.LOOKUP_IS_GROUP] == is_group}

    def GetSubCommandHelp(self):
        """Returns the subcommand help dict indexed by subcommand."""
        return self._GetSubHelp(is_group=False)

    def GetSubGroupHelp(self):
        """Returns the subgroup help dict indexed by subgroup."""
        return self._GetSubHelp(is_group=True)

    def PrintFlagDefinition(self, flag, disable_header=False):
        """Prints a flags definition list item."""
        if isinstance(flag, dict):
            flag = Flag(flag)
        super(CliTreeMarkdownGenerator, self).PrintFlagDefinition(flag, disable_header=disable_header)

    def _ExpandHelpText(self, doc):
        """{...} references were done when the tree was generated."""
        return doc