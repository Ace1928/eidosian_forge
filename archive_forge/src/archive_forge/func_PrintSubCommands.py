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