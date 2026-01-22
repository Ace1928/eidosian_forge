from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
import re
import textwrap
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
class MarkdownGenerator(six.with_metaclass(abc.ABCMeta, object)):
    """Command help markdown document generator base class.

  Attributes:
    _buf: Output document stream.
    _capsule: The one line description string.
    _command_name: The dotted command name.
    _command_path: The command path list.
    _doc: The output markdown document string.
    _docstring: The command docstring.
    _file_name: The command path name (used to name documents).
    _final_sections: The list of PrintFinalSections section names.
    _is_hidden: The command is hidden.
    _out: Output writer.
    _printed_sections: The set of already printed sections.
    _release_track: The calliope.base.ReleaseTrack.
  """

    def __init__(self, command_path, release_track, is_hidden):
        """Constructor.

    Args:
      command_path: The command path list.
      release_track: The base.ReleaseTrack of the command.
      is_hidden: The command is hidden if True.
    """
        self._command_path = command_path
        self._command_name = ' '.join(self._command_path)
        self._subcommands = None
        self._subgroups = None
        self._sort_top_level_args = None
        self._top = self._command_path[0] if self._command_path else ''
        self._buf = io.StringIO()
        self._out = self._buf.write
        self._capsule = ''
        self._docstring = ''
        self._final_sections = ['EXAMPLES', 'SEE ALSO']
        self._arg_sections = None
        self._sections = {}
        self._file_name = '_'.join(self._command_path)
        self._global_flags = set()
        self._is_hidden = is_hidden
        self._release_track = release_track
        self._printed_sections = set()

    @abc.abstractmethod
    def IsValidSubPath(self, sub_command_path):
        """Determines if the given sub command path is valid from this node.

    Args:
      sub_command_path: [str], The pieces of the command path.

    Returns:
      True, if the given path parts exist under this command or group node.
      False, if the sub path does not lead to a valid command or group.
    """
        pass

    @abc.abstractmethod
    def GetArguments(self):
        """Returns the command arguments."""
        pass

    def FormatExample(self, cmd, args, with_args):
        """Creates a link to the command reference from a command example.

    If with_args is False and the provided command includes args,
    returns None.

    Args:
      cmd: [str], a command.
      args: [str], args with the command.
      with_args: bool, whether the example is valid if it has args.

    Returns:
      (str) a representation of the command with a link to the reference, plus
      any args. | None, if the command isn't valid.
    """
        if args and (not with_args):
            return None
        ref = '/'.join(cmd)
        command_link = 'link:' + ref + '[' + ' '.join(cmd) + ']'
        if args:
            command_link += ' ' + ' '.join(args)
        return command_link

    @property
    def is_root(self):
        """Determine if this node should be treated as a "root" of the CLI tree.

    The top element is the root, but we also treat any additional release tracks
    as a root so that global flags are shown there as well.

    Returns:
      True if this node should be treated as a root, False otherwise.
    """
        if len(self._command_path) == 1:
            return True
        elif len(self._command_path) == 2:
            tracks = [t.prefix for t in base.ReleaseTrack.AllValues()]
            if self._command_path[-1] in tracks:
                return True
        return False

    @property
    def is_group(self):
        """Returns True if this node is a command group."""
        return bool(self._subgroups or self._subcommands)

    @property
    def sort_top_level_args(self):
        """Returns whether to sort the top level arguments in markdown docs."""
        return self._sort_top_level_args

    @property
    def is_topic(self):
        """Returns True if this node is a topic command."""
        if len(self._command_path) >= 3 and self._command_path[1] == self._release_track.prefix:
            command_index = 2
        else:
            command_index = 1
        return len(self._command_path) >= command_index + 1 and self._command_path[command_index] == 'topic'

    def _ExpandHelpText(self, text):
        """Expand command {...} references in text.

    Args:
      text: The text chunk to expand.

    Returns:
      The expanded help text.
    """
        return console_io.LazyFormat(text or '', command=self._command_name, man_name=self._file_name, top_command=self._top, parent_command=' '.join(self._command_path[:-1]), grandparent_command=' '.join(self._command_path[:-2]), index=self._capsule, **self._sections)

    def _SetArgSections(self):
        """Sets self._arg_sections in document order."""
        if self._arg_sections is None:
            self._arg_sections, self._global_flags = usage_text.GetArgSections(self.GetArguments(), self.is_root, self.is_group, self.sort_top_level_args)

    def _SplitCommandFromArgs(self, cmd):
        """Splits cmd into command and args lists.

    The command list part is a valid command and the args list part is the
    trailing args.

    Args:
      cmd: [str], A command + args list.

    Returns:
      (command, args): The command and args lists.
    """
        if len(cmd) <= 1:
            return (cmd, [])
        skip = 1
        i = skip
        while i <= len(cmd):
            i += 1
            if not self.IsValidSubPath(cmd[skip:i]):
                i -= 1
                break
        return (cmd[:i], cmd[i:])

    def _UserInput(self, msg):
        """Returns msg with user input markdown.

    Args:
      msg: str, The user input string.

    Returns:
      The msg string with embedded user input markdown.
    """
        return base.MARKDOWN_CODE + base.MARKDOWN_ITALIC + msg + base.MARKDOWN_ITALIC + base.MARKDOWN_CODE

    def _ArgTypeName(self, arg):
        """Returns the argument type name for arg."""
        return 'positional' if arg.is_positional else 'flag'

    def PrintSectionHeader(self, name, sep=True):
        """Prints the section header markdown for name.

    Args:
      name: str, The manpage section name.
      sep: boolean, Add trailing newline.
    """
        self._printed_sections.add(name)
        self._out('\n\n## {name}\n'.format(name=name))
        if sep:
            self._out('\n')

    def PrintUniverseInformationSection(self, disable_header=False):
        """Prints the command line information section.

    The information section provides disclaimer information on whether a command
    is available in a particular universe domain.

    Args:
      disable_header: Disable printing the section header if True.
    """
        if properties.IsDefaultUniverse():
            return
        if not disable_header:
            self.PrintSectionHeader('INFORMATION')
        code = base.MARKDOWN_CODE
        em = base.MARKDOWN_ITALIC
        if self._command.IsUniverseCompatible():
            info_body = f'{code}{self._command_name}{code} is supported in universe domain {em}{properties.GetUniverseDomain()}{em}; however, some of the values used in the help text may not be available. Command examples may not work as-is and may requires changes before execution.'
        else:
            info_body = f'{code}{self._command_name}{code} is not available in universe domain {em}{properties.GetUniverseDomain()}{em}.'
        self._out(info_body)
        self.PrintSectionIfExists('UNIVERSE ADDITIONAL INFO')

    def PrintNameSection(self, disable_header=False):
        """Prints the command line name section.

    Args:
      disable_header: Disable printing the section header if True.
    """
        if not disable_header:
            self.PrintSectionHeader('NAME')
        self._out('{command} - {index}\n'.format(command=self._command_name, index=_GetIndexFromCapsule(self._capsule)))

    def PrintSynopsisSection(self, disable_header=False):
        """Prints the command line synopsis section.

    Args:
      disable_header: Disable printing the section header if True.
    """
        if self.is_topic:
            return
        self._SetArgSections()
        code = base.MARKDOWN_CODE
        em = base.MARKDOWN_ITALIC
        if not disable_header:
            self.PrintSectionHeader('SYNOPSIS')
        self._out('{code}{command}{code}'.format(code=code, command=self._command_name))
        if self._subcommands and self._subgroups:
            self._out(' ' + em + 'GROUP' + em + ' | ' + em + 'COMMAND' + em)
        elif self._subcommands:
            self._out(' ' + em + 'COMMAND' + em)
        elif self._subgroups:
            self._out(' ' + em + 'GROUP' + em)
        remainder_usage = []
        for section in self._arg_sections:
            self._out(' ')
            self._out(usage_text.GetArgUsage(section.args, markdown=True, top=True, remainder_usage=remainder_usage))
        if self._global_flags:
            self._out(' [' + em + self._top.upper() + '_WIDE_FLAG ...' + em + ']')
        if remainder_usage:
            self._out(' ')
            self._out(' '.join(remainder_usage))
        self._out('\n')

    def _PrintArgDefinition(self, arg, depth=0, single=False):
        """Prints a positional or flag arg definition list item at depth."""
        usage = usage_text.GetArgUsage(arg, definition=True, markdown=True)
        if not usage:
            return
        self._out('\n{usage}{depth}\n'.format(usage=usage, depth=':' * (depth + _SECOND_LINE_OFFSET)))
        if arg.is_required and depth and (not single):
            modal = '\n+\nThis {arg_type} argument must be specified if any of the other arguments in this group are specified.'.format(arg_type=self._ArgTypeName(arg))
        else:
            modal = ''
        details = self.GetArgDetails(arg, depth=depth).replace('\n\n', '\n+\n')
        self._out('\n{details}{modal}\n'.format(details=details, modal=modal))

    def _PrintArgGroup(self, arg, depth=0, single=False):
        """Prints an arg group definition list at depth."""
        args = sorted(arg.arguments, key=usage_text.GetArgSortKey) if arg.sort_args else arg.arguments
        heading = []
        if arg.help or arg.is_mutex or arg.is_required:
            if arg.help:
                heading.append(arg.help)
            if arg.disable_default_heading:
                pass
            elif len(args) == 1 or args[0].is_required:
                if arg.is_required:
                    heading.append('This must be specified.')
            elif arg.is_mutex:
                if arg.is_required:
                    heading.append('Exactly one of these must be specified:')
                else:
                    heading.append('At most one of these can be specified:')
            elif arg.is_required:
                heading.append('At least one of these must be specified:')
        if not arg.is_hidden and heading:
            self._out('\n{0} {1}\n\n'.format(':' * (depth + _SECOND_LINE_OFFSET), '\n+\n'.join(heading)).replace('\n\n', '\n+\n'))
            heading = None
            depth += 1
        for a in args:
            if a.is_hidden:
                continue
            if a.is_group:
                single = False
                singleton = usage_text.GetSingleton(a)
                if singleton:
                    if not a.help:
                        a = singleton
                    else:
                        single = True
            if a.is_group:
                self._PrintArgGroup(a, depth=depth, single=single)
            else:
                self._PrintArgDefinition(a, depth=depth, single=single)

    def PrintPositionalDefinition(self, arg, depth=0):
        self._out('\n{usage}{depth}\n'.format(usage=usage_text.GetPositionalUsage(arg, markdown=True), depth=':' * (depth + _SECOND_LINE_OFFSET)))
        self._out('\n{arghelp}\n'.format(arghelp=self.GetArgDetails(arg)))

    def PrintFlagDefinition(self, flag, disable_header=False, depth=0):
        """Prints a flags definition list item.

    Args:
      flag: The flag object to display.
      disable_header: Disable printing the section header if True.
      depth: The indentation depth at which to print arg help text.
    """
        if not disable_header:
            self._out('\n')
        self._out('{usage}{depth}\n'.format(usage=usage_text.GetFlagUsage(flag, markdown=True), depth=':' * (depth + _SECOND_LINE_OFFSET)))
        self._out('\n{arghelp}\n'.format(arghelp=self.GetArgDetails(flag)))

    def PrintFlagSection(self, heading, arg, disable_header=False):
        """Prints a flag section.

    Args:
      heading: The flag section heading name.
      arg: The flag args / group.
      disable_header: Disable printing the section header if True.
    """
        if not disable_header:
            self.PrintSectionHeader(heading, sep=False)
        self._PrintArgGroup(arg)

    def PrintPositionalsAndFlagsSections(self, disable_header=False):
        """Prints the positionals and flags sections.

    Args:
      disable_header: Disable printing the section header if True.
    """
        if self.is_topic:
            return
        self._SetArgSections()
        for section in self._arg_sections:
            self.PrintFlagSection(section.heading, section.args, disable_header=disable_header)
        if self._global_flags:
            if not disable_header:
                self.PrintSectionHeader('{} WIDE FLAGS'.format(self._top.upper()), sep=False)
            self._out('\nThese flags are available to all commands: {}.\n\nRun *$ {} help* for details.\n'.format(', '.join(sorted(self._global_flags)), self._top))

    def PrintSubGroups(self, disable_header=False):
        """Prints the subgroup section if there are subgroups.

    Args:
      disable_header: Disable printing the section header if True.
    """
        if self._subgroups:
            self.PrintCommandSection('GROUP', self._subgroups, disable_header=disable_header)

    def PrintSubCommands(self, disable_header=False):
        """Prints the subcommand section if there are subcommands.

    Args:
      disable_header: Disable printing the section header if True.
    """
        if self._subcommands:
            if self.is_topic:
                self.PrintCommandSection('TOPIC', self._subcommands, is_topic=True, disable_header=disable_header)
            else:
                self.PrintCommandSection('COMMAND', self._subcommands, disable_header=disable_header)

    def PrintSectionIfExists(self, name, default=None, disable_header=False):
        """Print a section name if it exists.

    Args:
      name: str, The manpage section name.
      default: str, Default help_stuff if section name is not defined.
      disable_header: Disable printing the section header if True.
    """
        if name in self._printed_sections:
            return
        help_stuff = self._sections.get(name, default)
        if not help_stuff:
            return
        if callable(help_stuff):
            help_message = help_stuff()
        else:
            help_message = help_stuff
        if not disable_header:
            self.PrintSectionHeader(name)
        self._out('{message}\n'.format(message=textwrap.dedent(help_message).strip()))

    def PrintExtraSections(self, disable_header=False):
        """Print extra sections not in excluded_sections.

    Extra sections are sections that have not been printed yet.
    PrintSectionIfExists() skips sections that have already been printed.

    Args:
      disable_header: Disable printing the section header if True.
    """
        excluded_sections = set(self._final_sections + ['NOTES', 'UNIVERSE ADDITIONAL INFO'])
        for section in sorted(self._sections):
            if section.isupper() and section not in excluded_sections:
                self.PrintSectionIfExists(section, disable_header=disable_header)

    def PrintFinalSections(self, disable_header=False):
        """Print the final sections in order.

    Args:
      disable_header: Disable printing the section header if True.
    """
        for section in self._final_sections:
            self.PrintSectionIfExists(section, disable_header=disable_header)
        self.PrintNotesSection(disable_header=disable_header)

    def PrintCommandSection(self, name, subcommands, is_topic=False, disable_header=False):
        """Prints a group or command section.

    Args:
      name: str, The section name singular form.
      subcommands: dict, The subcommand dict.
      is_topic: bool, True if this is a TOPIC subsection.
      disable_header: Disable printing the section header if True.
    """
        content = ''
        for subcommand, help_info in sorted(six.iteritems(subcommands)):
            if self._is_hidden or not help_info.is_hidden:
                content += '\n*link:{ref}[{cmd}]*::\n\n{txt}\n'.format(ref='/'.join(self._command_path + [subcommand]), cmd=subcommand, txt=help_info.help_text)
        if content:
            if not disable_header:
                self.PrintSectionHeader(name + 'S')
            if is_topic:
                self._out('The supplementary help topics are:\n')
            else:
                self._out('{cmd} is one of the following:\n'.format(cmd=self._UserInput(name)))
            self._out(content)

    def GetNotes(self):
        """Returns the explicit NOTES section contents."""
        return self._sections.get('NOTES')

    def PrintNotesSection(self, disable_header=False):
        """Prints the NOTES section if needed.

    Args:
      disable_header: Disable printing the section header if True.
    """
        notes = self.GetNotes()
        if notes:
            if not disable_header:
                self.PrintSectionHeader('NOTES')
            if notes:
                self._out(notes + '\n\n')

    def GetArgDetails(self, arg, depth=0):
        """Returns the detailed help message for the given arg."""
        if getattr(arg, 'detailed_help', None):
            raise ValueError('{}: Use add_argument(help=...) instead of detailed_help="""{}""".'.format(self._command_name, getattr(arg, 'detailed_help')))
        return usage_text.GetArgDetails(arg, depth=depth)

    def _ExpandFormatReferences(self, doc):
        """Expand {...} references in doc."""
        doc = self._ExpandHelpText(doc)
        doc = NormalizeExampleSection(doc)
        pat = re.compile('^ *(\\$ .{%d,})$' % (_SPLIT - _FIRST_INDENT - _SECTION_INDENT), re.M)
        pos = 0
        rep = ''
        while True:
            match = pat.search(doc, pos)
            if not match:
                break
            rep += doc[pos:match.start(1)] + ExampleCommandLineSplitter().Split(doc[match.start(1):match.end(1)])
            pos = match.end(1)
        if rep:
            doc = rep + doc[pos:]
        return doc

    def _IsNotThisCommand(self, cmd):
        return '.'.join(cmd) != '.'.join(self._command_path)

    def _LinkMarkdown(self, doc, pat, with_args=True):
        """Build a representation of a doc, finding all command examples.

    Finds examples of both inline commands and commands on their own line.

    Args:
      doc: str, the doc to find examples in.
      pat: the compiled regexp pattern to match against (the "command" match
          group).
      with_args: bool, whether the examples are valid if they also have
          args.

    Returns:
      (str) The final representation of the doc.
    """
        pos = 0
        rep = ''
        while True:
            match = pat.search(doc, pos)
            if not match:
                break
            cmd, args = self._SplitCommandFromArgs(match.group('command').split(' '))
            lnk = self.FormatExample(cmd, args, with_args=with_args)
            if self._IsNotThisCommand(cmd) and lnk:
                rep += doc[pos:match.start('command')] + lnk
            else:
                rep += doc[pos:match.end('command')]
            rep += doc[match.end('command'):match.end('end')]
            pos = match.end('end')
        if rep:
            doc = rep + doc[pos:]
        return doc

    def InlineCommandExamplePattern(self):
        """Regex to search for inline command examples enclosed in ` or *.

    Contains a 'command' group and an 'end' group which will be used
    by the regexp search later.

    Returns:
      (str) the regex pattern, including a format string for the 'top'
      command.
    """
        return '(?<!\\n\\n)(?<!\\*\\(ALPHA\\)\\* )(?<!\\*\\(BETA\\)\\* )([`*])(?P<command>{top}( [a-z][-a-z0-9]*)*)(?P<end>\\1)'.format(top=re.escape(self._top))

    def _AddCommandLinkMarkdown(self, doc):
        """Add ([`*])command ...\\1 link markdown to doc."""
        if not self._command_path:
            return doc
        pat = re.compile(self.InlineCommandExamplePattern())
        doc = self._LinkMarkdown(doc, pat, with_args=False)
        return doc

    def CommandLineExamplePattern(self):
        """Regex to search for command examples starting with '$ '.

    Contains a 'command' group and an 'end' group which will be used
    by the regexp search later.

    Returns:
      (str) the regex pattern, including a format string for the 'top'
      command.
    """
        return '\\$ (?P<end>(?P<command>{top}((?: (?!(example|my|sample)-)[a-z][-a-z0-9]*)*))).?[ `\\n]'.format(top=re.escape(self._top))

    def _AddCommandLineLinkMarkdown(self, doc):
        """Add $ command ... link markdown to doc."""
        if not self._command_path:
            return doc
        pat = re.compile(self.CommandLineExamplePattern())
        doc = self._LinkMarkdown(doc, pat, with_args=True)
        return doc

    def _AddManPageLinkMarkdown(self, doc):
        """Add <top> ...(1) man page link markdown to doc."""
        if not self._command_path:
            return doc
        pat = re.compile('(\\*?(' + self._top + '(?:[-_ a-z])*)\\*?)\\(1\\)')
        pos = 0
        rep = ''
        while True:
            match = pat.search(doc, pos)
            if not match:
                break
            cmd = match.group(2).replace('_', ' ')
            ref = cmd.replace(' ', '/')
            lnk = '*link:' + ref + '[' + cmd + ']*'
            rep += doc[pos:match.start(2)] + lnk
            pos = match.end(1)
        if rep:
            doc = rep + doc[pos:]
        return doc

    def _FixAirQuotesMarkdown(self, doc):
        """Change ``.*[[:alnum:]]{2,}.*'' quotes => _UserInput(*) in doc."""
        pat = re.compile("[^`](``([^`']*)'')")
        pos = 0
        rep = ''
        for match in pat.finditer(doc):
            if re.search('\\w\\w', match.group(2)):
                quoted_string = self._UserInput(match.group(2))
            else:
                quoted_string = match.group(1)
            rep += doc[pos:match.start(1)] + quoted_string
            pos = match.end(1)
        if rep:
            doc = rep + doc[pos:]
        return doc

    def _IsUniverseCompatible(self):
        return not properties.IsDefaultUniverse() and (not isinstance(self._command, dict)) and self._command.IsUniverseCompatible()

    def _ReplaceGDULinksWithUniverseLinks(self, doc):
        """Replace static GDU Links with Universe Links."""
        if self._IsUniverseCompatible():
            doc = re.sub('cloud.google.com', properties.GetUniverseDocumentDomain(), doc)
        return doc

    def Edit(self, doc=None):
        """Applies edits to a copy of the generated markdown in doc.

    The sub-edit method call order might be significant. This method allows
    the combined edits to be tested without relying on the order.

    Args:
      doc: The markdown document string to edit, None for the output buffer.

    Returns:
      An edited copy of the generated markdown.
    """
        if doc is None:
            doc = self._buf.getvalue()
        doc = self._ExpandFormatReferences(doc)
        doc = self._AddCommandLineLinkMarkdown(doc)
        doc = self._AddCommandLinkMarkdown(doc)
        doc = self._AddManPageLinkMarkdown(doc)
        doc = self._FixAirQuotesMarkdown(doc)
        doc = self._ReplaceGDULinksWithUniverseLinks(doc)
        return doc

    def Generate(self):
        """Generates markdown for the command, group or topic, into a string.

    Returns:
      An edited copy of the generated markdown.
    """
        self._out('# {0}(1)\n'.format(self._file_name.upper()))
        self.PrintUniverseInformationSection()
        self.PrintNameSection()
        self.PrintSynopsisSection()
        self.PrintSectionIfExists('DESCRIPTION')
        self.PrintSectionIfExists('EXAMPLES')
        self.PrintPositionalsAndFlagsSections()
        self.PrintSubGroups()
        self.PrintSubCommands()
        self.PrintExtraSections()
        self.PrintFinalSections()
        return self.Edit()