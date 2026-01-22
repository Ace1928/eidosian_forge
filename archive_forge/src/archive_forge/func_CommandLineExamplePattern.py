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
def CommandLineExamplePattern(self):
    """Regex to search for command examples starting with '$ '.

    Contains a 'command' group and an 'end' group which will be used
    by the regexp search later.

    Returns:
      (str) the regex pattern, including a format string for the 'top'
      command.
    """
    return '\\$ (?P<end>(?P<command>{top}((?: (?!(example|my|sample)-)[a-z][-a-z0-9]*)*))).?[ `\\n]'.format(top=re.escape(self._top))