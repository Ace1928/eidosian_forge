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
def CopyAllSubElementsTo(self, other_group, ignore):
    """Copies all the sub groups and commands from this group to the other.

    Args:
      other_group: CommandGroup, The other group to populate.
      ignore: set(str), Names of elements not to copy.
    """
    other_group._groups_to_load.update({name: impl_paths for name, impl_paths in six.iteritems(self._groups_to_load) if name not in ignore})
    other_group._commands_to_load.update({name: impl_paths for name, impl_paths in six.iteritems(self._commands_to_load) if name not in ignore})