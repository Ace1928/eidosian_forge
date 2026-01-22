from a man-ish style runtime document.
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import json
import os
import re
import shlex
import subprocess
import tarfile
import textwrap
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.command_lib.static_completion import generate as generate_static
from googlecloudsdk.command_lib.static_completion import lookup
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
from six.moves import range
class ManPageCliTreeGenerator(CliTreeGenerator):
    """man page CLI tree generator."""

    @classmethod
    def _GetManPageCollectorType(cls):
        """Returns the man page collector type."""
        if files.FindExecutableOnPath('man'):
            return _ManCommandCollector
        return _ManUrlCollector

    def __init__(self, command_name):
        super(ManPageCliTreeGenerator, self).__init__(command_name)
        self.collector_type = self._GetManPageCollectorType()

    def GetVersion(self):
        """Returns the CLI_VERSION string."""
        if not self.collector_type:
            return cli_tree.CLI_VERSION_UNKNOWN
        return self.collector_type.GetVersion()

    def AddFlags(self, command, content, is_global=False):
        """Adds flags in content lines to command."""

        def _NameTypeValueNargs(name, type_=None, value=None, nargs=None):
            """Returns the (name, type, value-metavar, nargs) for flag name."""
            if name.startswith('--'):
                if '=' in name:
                    name, value = name.split('=', 1)
                    if name[-1] == '[':
                        name = name[:-1]
                        if value.endswith(']'):
                            value = value[:-1]
                        nargs = '?'
                    else:
                        nargs = '1'
                    type_ = 'string'
            elif len(name) > 2:
                value = name[2:]
                if value[0].isspace():
                    value = value[1:]
                if value.startswith('['):
                    value = value[1:]
                    if value.endswith(']'):
                        value = value[:-1]
                    nargs = '?'
                else:
                    nargs = '1'
                name = name[:2]
                type_ = 'string'
            if type_ is None or value is None or nargs is None:
                type_ = 'bool'
                value = ''
                nargs = '0'
            return (name, type_, value, nargs)

        def _Add(name, description, category, type_, value, nargs):
            """Adds a flag."""
            name, type_, value, nargs = _NameTypeValueNargs(name, type_, value, nargs)
            default = ''
            command[cli_tree.LOOKUP_FLAGS][name] = _Flag(name=name, description='\n'.join(description), type_=type_, value=value, nargs=nargs, category=category, default=default, is_required=False, is_global=is_global)

        def _AddNames(names, description, category):
            """Add a flag name list."""
            if names:
                _, type_, value, nargs = _NameTypeValueNargs(names[-1])
                for name in names:
                    _Add(name, description, category, type_, value, nargs)
        names = []
        description = []
        category = ''
        for line in content:
            if line.lstrip().startswith('-'):
                _AddNames(names, description, category)
                line = line.lstrip()
                names = line.strip().replace(', -', ', --').split(', -')
                if ' ' in names[-1]:
                    names[-1], text = names[-1].split(' ', 1)
                    description = [text.strip()]
                else:
                    description = []
            elif line.startswith('### '):
                category = line[4:]
            else:
                description.append(line)
        _AddNames(names, description, category)

    def _Generate(self, path, collector):
        """Generates and returns the CLI subtree rooted at path."""
        command = _Command(path)
        while True:
            heading, content = collector.Collect()
            if not heading:
                break
            elif heading == 'NAME':
                if content:
                    command[cli_tree.LOOKUP_CAPSULE] = re.sub('.* -+  *', '', content[0]).strip()
            elif heading == 'FLAGS':
                self.AddFlags(command, content)
            elif heading not in ('BUGS', 'COLOPHON', 'COMPATIBILITY', 'HISTORY', 'STANDARDS', 'SYNOPSIS'):
                blocks = []
                begin = 0
                end = 0
                while end < len(content):
                    if content[end].startswith('###'):
                        if begin < end:
                            blocks.append(_NormalizeSpace('\n'.join(content[begin:end])))
                        blocks.append(content[end])
                        begin = end + 1
                    end += 1
                if begin < end:
                    blocks.append(_NormalizeSpace('\n'.join(content[begin:end])))
                text = '\n'.join(blocks)
                if heading in command[cli_tree.LOOKUP_SECTIONS]:
                    command[cli_tree.LOOKUP_SECTIONS][heading] += '\n\n' + text
                else:
                    command[cli_tree.LOOKUP_SECTIONS][heading] = text
        return command

    def Generate(self):
        """Generates and returns the CLI tree rooted at self.command_name."""
        if not self.collector_type:
            return None
        collector = self.collector_type(self.command_name)
        if not collector:
            return None
        tree = self._Generate([self.command_name], collector)
        if not tree:
            return None
        tree[cli_tree.LOOKUP_CLI_VERSION] = self.GetVersion()
        tree[cli_tree.LOOKUP_VERSION] = cli_tree.VERSION
        return tree