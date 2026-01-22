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
def InlineCommandExamplePattern(self):
    """Regex to search for inline command examples enclosed in ` or *.

    Contains a 'command' group and an 'end' group which will be used
    by the regexp search later.

    Returns:
      (str) the regex pattern, including a format string for the 'top'
      command.
    """
    return '(?<!\\n\\n)(?<!\\*\\(ALPHA\\)\\* )(?<!\\*\\(BETA\\)\\* )([`*])(?P<command>{top}( [a-z][-a-z0-9]*)*)(?P<end>\\1)'.format(top=re.escape(self._top))