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
def PrintSubGroups(self, disable_header=False):
    """Prints the subgroup section if there are subgroups.

    Args:
      disable_header: Disable printing the section header if True.
    """
    if self._subgroups:
        self.PrintCommandSection('GROUP', self._subgroups, disable_header=disable_header)