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
def _ConvertParagraph(self, i):
    """Detects and converts + markdown line (length 1).

    Emits a blank line but retains the current indent.

    Args:
      i: The current character index in self._line.

    Returns:
      -1 if the input line is a '+' markdown, i otherwise.
    """
    if len(self._line) != 1 or self._line[0] != '+':
        return i
    self._Fill()
    self._lists[self._depth].line_break_seen = True
    if self._lists[self._depth].bullet:
        self._renderer.List(self._depth - 1, end=True)
        if self._depth:
            self._depth -= 1
        else:
            self._lists[self._depth].bullet = False
    self._renderer.Line()
    self._next_paragraph = True
    return -1