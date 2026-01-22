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
def _ConvertExample(self, i):
    """Detects and converts an example markdown line.

    Example lines are indented by one or more space characters.

    Args:
      i: The current character index in self._line.

    Returns:
      -1 if the input line is is an example line markdown, i otherwise.
    """
    example_allowed = False if self._lists[self._depth].bullet else True
    if not self._in_example_section:
        example_allowed = example_allowed and (not self._buf.strip())
    if not i or (not example_allowed and (not (self._example or self._paragraph))):
        return i
    self._Example(i)
    return -1