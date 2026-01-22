from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import renderer
from prompt_toolkit.token import Token
def TableLine(self, line, indent=0):
    """Adds an indented table line to the output.

    Args:
      line: The line to add. A newline will be added.
      indent: The number of characters to indent the table.
    """
    self._AddToken(indent * ' ' + line)
    self._NewLine()