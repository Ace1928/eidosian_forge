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
def _AnchorStyle2(self, buf, i):
    """Checks for [text](target) hyperlink anchor markdown.

    Hyperlink anchors are of the form:
      '[' <text> ']' '(' <target> ')'
    For example:
      [Google Search](http://www.google.com)
      [](http://www.show.the.link)
    The underlying renderer determines how the parts are displayed.

    Args:
      buf: Input buffer.
      i: The buf[] index of ':'.

    Returns:
      (i, target, text)
        i: The buf[] index just past the link, 0 if no link.
        target: The link target.
        text: The link text.
    """
    text_beg = i + 1
    text_end = _GetNestedGroup(buf, i, '[', ']')
    if not text_end or text_end >= len(buf) - 1 or buf[text_end + 1] != '(':
        return (0, None, None)
    target_beg = text_end + 2
    target_end = _GetNestedGroup(buf, target_beg - 1, '(', ')')
    if not target_end or target_end <= target_beg:
        return (0, None, None)
    return (target_end + 1, buf[target_beg:target_end], buf[text_beg:text_end])