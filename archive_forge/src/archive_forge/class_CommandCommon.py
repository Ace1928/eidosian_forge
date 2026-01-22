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
class CommandCommon(object):
    """A base class for CommandGroup and Command.

  It is responsible for extracting arguments from the modules and does argument
  validation, since this is always the same for groups and commands.
  """

    def __init__(self, common_type, path, release_track, cli_generator, parser_group, allow_positional_args, parent_group):
        """Create a new CommandCommon.

    Args:
      common_type: base._Common, The actual loaded user written command or
        group class.
      path: [str], A list of group names that got us down to this command group
        with respect to the CLI itself.  This path should be used for things
        like error reporting when a specific element in the tree needs to be
        referenced.
      release_track: base.ReleaseTrack, The release track (ga, beta, alpha,
        preview) that this command group is in.  This will apply to all commands
        under it.
      cli_generator: cli.CLILoader, The builder used to generate this CLI.
      parser_group: argparse.Parser, The parser that this command or group will
        live in.
      allow_positional_args: bool, True if this command can have positional
        arguments.
      parent_group: CommandGroup, The parent of this command or group. None if
        at the root.
    """
        self.category = common_type.category
        self._parent_group = parent_group
        self.name = path[-1]
        self.cli_name = self.name.replace('_', '-')
        log.debug('Loaded Command Group: %s', path)
        path[-1] = self.cli_name
        self._path = path
        self.dotted_name = '.'.join(path)
        self._cli_generator = cli_generator
        self._common_type = common_type
        self._common_type._cli_generator = cli_generator
        self._common_type._release_track = release_track
        self.is_group = any([t == base.Group for t in common_type.__mro__])
        if parent_group:
            if parent_group.IsHidden():
                self._common_type._is_hidden = True
            if parent_group.IsUniverseCompatible() and self._common_type._universe_compatible is None:
                self._common_type._universe_compatible = True
            if parent_group.IsUnicodeSupported():
                self._common_type._is_unicode_supported = True
            if parent_group.Notices():
                for tag, msg in six.iteritems(parent_group.Notices()):
                    self._common_type.AddNotice(tag, msg, preserve_existing=True)
        self.detailed_help = getattr(self._common_type, 'detailed_help', {})
        self._ExtractHelpStrings(self._common_type.__doc__)
        self._AssignParser(parser_group=parser_group, allow_positional_args=allow_positional_args)

    def Notices(self):
        """Gets the notices of this command or group."""
        return self._common_type.Notices()

    def ReleaseTrack(self):
        """Gets the release track of this command or group."""
        return self._common_type.ReleaseTrack()

    def IsHidden(self):
        """Gets the hidden status of this command or group."""
        return self._common_type.IsHidden()

    def IsUniverseCompatible(self):
        """Gets the universe compatible status of this command or group."""
        return self._common_type.IsUniverseCompatible()

    def IsUnicodeSupported(self):
        """Gets the unicode supported status of this command or group."""
        return self._common_type.IsUnicodeSupported()

    def IsRoot(self):
        """Returns True if this is the root element in the CLI tree."""
        return not self._parent_group

    def _TopCLIElement(self):
        """Gets the top group of this CLI."""
        if self.IsRoot():
            return self
        return self._parent_group._TopCLIElement()

    def _ExtractHelpStrings(self, docstring):
        """Extracts short help, long help and man page index from a docstring.

    Sets self.short_help, self.long_help and self.index_help and adds release
    track tags if needed.

    Args:
      docstring: The docstring from which short and long help are to be taken
    """
        self.short_help, self.long_help = usage_text.ExtractHelpStrings(docstring)
        if 'brief' in self.detailed_help:
            self.short_help = re.sub('\\s', ' ', self.detailed_help['brief']).strip()
        if self.short_help and (not self.short_help.endswith('.')):
            self.short_help += '.'
        if self.Notices():
            all_notices = '\n\n' + '\n\n'.join(sorted(self.Notices().values())) + '\n\n'
            description = self.detailed_help.get('DESCRIPTION')
            if description:
                self.detailed_help = dict(self.detailed_help)
                self.detailed_help['DESCRIPTION'] = all_notices + textwrap.dedent(description)
            if self.short_help == self.long_help:
                self.long_help += all_notices
            else:
                self.long_help = self.short_help + all_notices + self.long_help
        self.index_help = self.short_help
        if len(self.index_help) > 1:
            if self.index_help[0].isupper() and (not self.index_help[1].isupper()):
                self.index_help = self.index_help[0].lower() + self.index_help[1:]
            if self.index_help[-1] == '.':
                self.index_help = self.index_help[:-1]
        tags = []
        tag = self.ReleaseTrack().help_tag
        if tag:
            tags.append(tag)
        if self.Notices():
            tags.extend(sorted(self.Notices().keys()))
        if tags:
            tag = ' '.join(tags) + ' '

            def _InsertTag(txt):
                return re.sub('^(\\s*)', '\\1' + tag, txt)
            self.short_help = _InsertTag(self.short_help)
            if not self.long_help.startswith('#'):
                self.long_help = _InsertTag(self.long_help)
            description = self.detailed_help.get('DESCRIPTION')
            if description and (not re.match('^[ \\n]*\\{(description|index)\\}', description)):
                self.detailed_help = dict(self.detailed_help)
                self.detailed_help['DESCRIPTION'] = _InsertTag(textwrap.dedent(description))

    def GetNotesHelpSection(self, contents=None):
        """Returns the NOTES section with explicit and generated help."""
        if not contents:
            contents = self.detailed_help.get('NOTES')
        notes = _Notes(contents)
        if self.IsHidden():
            notes.AddLine('This command is an internal implementation detail and may change or disappear without notice.')
        notes.AddLine(self.ReleaseTrack().help_note)
        alternates = self.GetExistingAlternativeReleaseTracks()
        if alternates:
            notes.AddLine('{} also available:'.format(text.Pluralize(len(alternates), 'This variant is', 'These variants are')))
            notes.AddLine('')
            for alternate in alternates:
                notes.AddLine('  $ ' + alternate)
                notes.AddLine('')
        return notes.GetContents()

    def _AssignParser(self, parser_group, allow_positional_args):
        """Assign a parser group to model this Command or CommandGroup.

    Args:
      parser_group: argparse._ArgumentGroup, the group that will model this
          command or group's arguments.
      allow_positional_args: bool, Whether to allow positional args for this
          group or not.

    """
        if not parser_group:
            self._parser = parser_extensions.ArgumentParser(description=self.long_help, add_help=False, prog=self.dotted_name, calliope_command=self)
        else:
            self._parser = parser_group.add_parser(self.cli_name, help=self.short_help, description=self.long_help, add_help=False, prog=self.dotted_name, calliope_command=self)
        self._sub_parser = None
        self.ai = parser_arguments.ArgumentInterceptor(parser=self._parser, is_global=not parser_group, cli_generator=self._cli_generator, allow_positional=allow_positional_args)
        self.ai.add_argument('-h', action=actions.ShortHelpAction(self), is_replicated=True, category=base.COMMONLY_USED_FLAGS, help='Print a summary help and exit.')
        self.ai.add_argument('--help', action=actions.RenderDocumentAction(self, '--help'), is_replicated=True, category=base.COMMONLY_USED_FLAGS, help='Display detailed help.')
        self.ai.add_argument('--document', action=actions.RenderDocumentAction(self), is_replicated=True, nargs=1, metavar='ATTRIBUTES', type=arg_parsers.ArgDict(), hidden=True, help='THIS TEXT SHOULD BE HIDDEN')
        self._AcquireArgs()

    def IsValidSubPath(self, command_path):
        """Determines if the given sub command path is valid from this node.

    Args:
      command_path: [str], The pieces of the command path.

    Returns:
      True, if the given path parts exist under this command or group node.
      False, if the sub path does not lead to a valid command or group.
    """
        current = self
        for part in command_path:
            current = current.LoadSubElement(part)
            if not current:
                return False
        return True

    def AllSubElements(self):
        """Gets all the sub elements of this group.

    Returns:
      set(str), The names of all sub groups or commands under this group.
    """
        return []

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
        return 0

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
        pass

    def LoadSubElementByPath(self, path):
        """Load a specific sub group or command by path.

    If path is empty, returns the current element.

    Args:
      path: list of str, The names of the elements to load down the hierarchy.

    Returns:
      _CommandCommon, The loaded sub element, or None if it did not exist.
    """
        curr = self
        for part in path:
            curr = curr.LoadSubElement(part)
            if curr is None:
                return None
        return curr

    def GetPath(self):
        return self._path

    def GetUsage(self):
        return usage_text.GetUsage(self, self.ai)

    def GetSubCommandHelps(self):
        return {}

    def GetSubGroupHelps(self):
        return {}

    def _AcquireArgs(self):
        """Calls the functions to register the arguments for this module."""
        self._common_type._Flags(self.ai)
        self._common_type.Args(self.ai)
        if self._parent_group:
            for arg in self._parent_group.ai.arguments:
                self.ai.arguments.append(arg)
            if self._parent_group.ai.concept_handler:
                if not self.ai.concept_handler:
                    self.ai.add_concepts(handlers.RuntimeHandler())
                for concept_details in self._parent_group.ai.concept_handler._all_concepts:
                    try:
                        self.ai.concept_handler.AddConcept(**concept_details)
                    except handlers.RepeatedConceptName:
                        raise parser_errors.ArgumentException('repeated concept in {command}: {concept_name}'.format(command=self.dotted_name, concept_name=concept_details['name']))
            for flag in self._parent_group.GetAllAvailableFlags():
                if flag.is_replicated:
                    continue
                if flag.do_not_propagate:
                    continue
                if flag.is_required:
                    continue
                try:
                    self.ai.AddFlagActionFromAncestors(flag)
                except argparse.ArgumentError:
                    raise parser_errors.ArgumentException('repeated flag in {command}: {flag}'.format(command=self.dotted_name, flag=flag.option_strings))
            self.ai.display_info.AddLowerDisplayInfo(self._parent_group.ai.display_info)

    def GetAllAvailableFlags(self, include_global=True, include_hidden=True):
        flags = self.ai.flag_args + self.ai.ancestor_flag_args
        if include_global and include_hidden:
            return flags
        return [f for f in flags if (include_global or not f.is_global) and (include_hidden or not f.is_hidden)]

    def GetSpecificFlags(self, include_hidden=True):
        flags = self.ai.flag_args
        if include_hidden:
            return flags
        return [f for f in flags if not f.hidden]

    def GetExistingAlternativeReleaseTracks(self, value=None):
        """Gets the names for the command in other release tracks.

    Args:
      value: str, Optional value being parsed after the command.

    Returns:
      [str]: The names for the command in other release tracks.
    """
        existing_alternatives = []
        path = self.GetPath()
        if value:
            path.append(value)
        alternates = self._cli_generator.ReplicateCommandPathForAllOtherTracks(path)
        if alternates:
            top_element = self._TopCLIElement()
            for _, command_path in sorted(six.iteritems(alternates), key=lambda x: x[0].prefix or ''):
                alternative_cmd = top_element.LoadSubElementByPath(command_path[1:])
                if alternative_cmd and (not alternative_cmd.IsHidden()):
                    existing_alternatives.append(' '.join(command_path))
        return existing_alternatives