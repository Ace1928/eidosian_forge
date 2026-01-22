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
def PrintPositionalDefinition(self, arg, depth=0):
    self._out('\n{usage}{depth}\n'.format(usage=usage_text.GetPositionalUsage(arg, markdown=True), depth=':' * (depth + _SECOND_LINE_OFFSET)))
    self._out('\n{arghelp}\n'.format(arghelp=self.GetArgDetails(arg)))