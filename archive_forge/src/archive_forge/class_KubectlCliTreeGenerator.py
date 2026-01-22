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
class KubectlCliTreeGenerator(CliTreeGenerator):
    """kubectl CLI tree generator."""

    def AddFlags(self, command, content, is_global=False):
        """Adds flags in content lines to command."""
        for line in content:
            flags, description = line.strip().split(':', 1)
            flag = flags.split(', ')[-1]
            name, value = flag.split('=')
            if value in ('true', 'false'):
                value = ''
                type_ = 'bool'
            else:
                value = 'VALUE'
                type_ = 'string'
            default = ''
            command[cli_tree.LOOKUP_FLAGS][name] = _Flag(name=name, description=description, type_=type_, value=value, default=default, is_required=False, is_global=is_global)

    def SubTree(self, path):
        """Generates and returns the CLI subtree rooted at path."""
        command = _Command(path)
        text = self.Run(path[1:] + ['--help'])
        collector = _KubectlCollector(text)
        while True:
            heading, content = collector.Collect()
            if not heading:
                break
            elif heading == 'COMMANDS':
                for line in content:
                    try:
                        name = line.split()[0]
                    except IndexError:
                        continue
                    command[cli_tree.LOOKUP_IS_GROUP] = True
                    command[cli_tree.LOOKUP_COMMANDS][name] = self.SubTree(path + [name])
            elif heading in ('DESCRIPTION', 'EXAMPLES'):
                command[cli_tree.LOOKUP_SECTIONS][heading] = '\n'.join(content)
            elif heading == 'FLAGS':
                self.AddFlags(command, content)
        return command

    def GetVersion(self):
        """Returns the CLI_VERSION string."""
        if not self._cli_version:
            try:
                verbose_version = self.Run(['version', '--client'])
                match = re.search('GitVersion:"([^"]*)"', verbose_version)
                self._cli_version = match.group(1)
            except:
                self._cli_version = cli_tree.CLI_VERSION_UNKNOWN
        return self._cli_version

    def Generate(self):
        """Generates and returns the CLI tree rooted at self.command_name."""
        tree = self.SubTree([self.command_name])
        text = self.Run(['options'])
        collector = _KubectlCollector(text)
        _, content = collector.Collect(strip_headings=True)
        content.append('  --help=true: List detailed command help.')
        self.AddFlags(tree, content, is_global=True)
        tree[cli_tree.LOOKUP_CLI_VERSION] = self.GetVersion()
        tree[cli_tree.LOOKUP_VERSION] = cli_tree.VERSION
        return tree