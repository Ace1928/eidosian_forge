from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import re
import sys
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.document_renderers import devsite_renderer
from googlecloudsdk.core.document_renderers import html_renderer
from googlecloudsdk.core.document_renderers import linter_renderer
from googlecloudsdk.core.document_renderers import man_renderer
from googlecloudsdk.core.document_renderers import markdown_renderer
from googlecloudsdk.core.document_renderers import renderer
from googlecloudsdk.core.document_renderers import text_renderer
def _ConvertRemainder(self, i):
    """Detects and converts any remaining markdown text.

    The input line is always consumed by this method. It should be the last
    _Convert*() method called for each input line.

    Args:
      i: The current character index in self._line.

    Returns:
      -1
    """
    self._lists[self._depth].line_break_seen = False
    self._buf += ' ' + self._line[i:]
    return -1