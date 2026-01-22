from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import re
import textwrap
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.calliope import display
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.calliope.concepts import handlers
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core.util import text
import six
class CommandGroup(CommandCommon):
    """A class to encapsulate a group of commands."""

    def __init__(self, impl_paths, path, release_track, construction_id, cli_generator, parser_group, parent_group=None, allow_empty=False):
        """Create a new command group.

    Args:
      impl_paths: [str], A list of file paths to the command implementation for
        this group.
      path: [str], A list of group names that got us down to this command group
        with respect to the CLI itself.  This path should be used for things
        like error reporting when a specific element in the tree needs to be
        referenced.
      release_track: base.ReleaseTrack, The release track (ga, beta, alpha) that
        this command group is in.  This will apply to all commands under it.
      construction_id: str, A unique identifier for the CLILoader that is
        being constructed.
      cli_generator: cli.CLILoader, The builder used to generate this CLI.
      parser_group: the current argparse parser, or None if this is the root
        command group.  The root command group will allocate the initial
        top level argparse parser.
      parent_group: CommandGroup, The parent of this group. None if at the
        root.
      allow_empty: bool, True to allow creating this group as empty to start
        with.

    Raises:
      LayoutException: if the module has no sub groups or commands
    """
        common_type = command_loading.LoadCommonType(impl_paths, path, release_track, construction_id, is_command=False)
        super(CommandGroup, self).__init__(common_type, path=path, release_track=release_track, cli_generator=cli_generator, allow_positional_args=False, parser_group=parser_group, parent_group=parent_group)
        self._construction_id = construction_id
        self.groups = {}
        self.commands = {}
        self._groups_to_load = {}
        self._commands_to_load = {}
        self._unloadable_elements = set()
        group_infos, command_infos = command_loading.FindSubElements(impl_paths, path)
        self._groups_to_load.update(group_infos)
        self._commands_to_load.update(command_infos)
        if not allow_empty and (not self._groups_to_load) and (not self._commands_to_load):
            raise command_loading.LayoutException('Group {0} has no subgroups or commands'.format(self.dotted_name))
        self.SubParser()

    def CopyAllSubElementsTo(self, other_group, ignore):
        """Copies all the sub groups and commands from this group to the other.

    Args:
      other_group: CommandGroup, The other group to populate.
      ignore: set(str), Names of elements not to copy.
    """
        other_group._groups_to_load.update({name: impl_paths for name, impl_paths in six.iteritems(self._groups_to_load) if name not in ignore})
        other_group._commands_to_load.update({name: impl_paths for name, impl_paths in six.iteritems(self._commands_to_load) if name not in ignore})

    def SubParser(self):
        """Gets or creates the argparse sub parser for this group.

    Returns:
      The argparse subparser that children of this group should register with.
          If a sub parser has not been allocated, it is created now.
    """
        if not self._sub_parser:
            self._sub_parser = self._parser.add_subparsers(action=parser_extensions.CommandGroupAction, calliope_command=self)
        return self._sub_parser

    def AllSubElements(self):
        """Gets all the sub elements of this group.

    Returns:
      set(str), The names of all sub groups or commands under this group.
    """
        return set(self._groups_to_load.keys()) | set(self._commands_to_load.keys())

    def IsValidSubElement(self, name):
        """Determines if the given name is a valid sub group or command.

    Args:
      name: str, The name of the possible sub element.

    Returns:
      bool, True if the name is a valid sub element of this group.
    """
        return bool(self.LoadSubElement(name))

    def LoadAllSubElements(self, recursive=False, ignore_load_errors=False):
        """Load all the sub groups and commands of this group.

    Args:
      recursive: bool, True to continue loading all sub groups, False, to just
        load the elements under the group.
      ignore_load_errors: bool, True to ignore command load failures. This
        should only be used when it is not critical that all data is returned,
        like for optimizations like static tab completion.

    Returns:
      int, The total number of elements loaded.
    """
        total = 0
        for name in self.AllSubElements():
            try:
                element = self.LoadSubElement(name)
                total += 1
            except:
                element = None
                if not ignore_load_errors:
                    raise
            if element and recursive:
                total += element.LoadAllSubElements(recursive=recursive, ignore_load_errors=ignore_load_errors)
        return total

    def LoadSubElement(self, name, allow_empty=False, release_track_override=None):
        """Load a specific sub group or command.

    Args:
      name: str, The name of the element to load.
      allow_empty: bool, True to allow creating this group as empty to start
        with.
      release_track_override: base.ReleaseTrack, Load the given sub-element
        under the given track instead of that of the parent. This should only
        be used when specifically creating the top level release track groups.

    Returns:
      _CommandCommon, The loaded sub element, or None if it did not exist.
    """
        name = name.replace('-', '_')
        existing = self.groups.get(name, None)
        if not existing:
            existing = self.commands.get(name, None)
        if existing:
            return existing
        if name in self._unloadable_elements:
            return None
        element = None
        try:
            if name in self._groups_to_load:
                element = CommandGroup(self._groups_to_load[name], self._path + [name], release_track_override or self.ReleaseTrack(), self._construction_id, self._cli_generator, self.SubParser(), parent_group=self, allow_empty=allow_empty)
                self.groups[element.name] = element
            elif name in self._commands_to_load:
                element = Command(self._commands_to_load[name], self._path + [name], release_track_override or self.ReleaseTrack(), self._construction_id, self._cli_generator, self.SubParser(), parent_group=self)
                self.commands[element.name] = element
        except command_loading.ReleaseTrackNotImplementedException as e:
            self._unloadable_elements.add(name)
            log.debug(e)
        return element

    def GetSubCommandHelps(self):
        return dict(((item.cli_name, usage_text.HelpInfo(help_text=item.short_help, is_hidden=item.IsHidden(), release_track=item.ReleaseTrack)) for item in self.commands.values()))

    def GetSubGroupHelps(self):
        return dict(((item.cli_name, usage_text.HelpInfo(help_text=item.short_help, is_hidden=item.IsHidden(), release_track=item.ReleaseTrack())) for item in self.groups.values()))

    def RunGroupFilter(self, context, args):
        """Constructs and runs the Filter() method of all parent groups.

    This recurses up to the root group and then constructs each group and runs
    its Filter() method down the tree.

    Args:
      context: {}, The context dictionary that Filter() can modify.
      args: The argparse namespace.
    """
        if self._parent_group:
            self._parent_group.RunGroupFilter(context, args)
        self._common_type().Filter(context, args)

    def GetCategoricalUsage(self):
        return usage_text.GetCategoricalUsage(self, self._GroupSubElementsByCategory())

    def GetUncategorizedUsage(self):
        return usage_text.GetUncategorizedUsage(self)

    def GetHelpHint(self):
        return usage_text.GetHelpHint(self)

    def _GroupSubElementsByCategory(self):
        """Returns dictionary mapping each category to its set of subelements."""

        def _GroupSubElementsOfSameTypeByCategory(elements):
            """Returns dictionary mapping specific to element type."""
            categorized_dict = collections.defaultdict(set)
            for element in elements.values():
                if not element.IsHidden():
                    if element.category:
                        categorized_dict[element.category].add(element)
                    else:
                        categorized_dict[base.UNCATEGORIZED_CATEGORY].add(element)
            return categorized_dict
        self.LoadAllSubElements()
        categories = {}
        categories['command'] = _GroupSubElementsOfSameTypeByCategory(self.commands)
        categories['command_group'] = _GroupSubElementsOfSameTypeByCategory(self.groups)
        return categories