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
def _ConvertHeading(self, i):
    """Detects and converts a markdown heading line.

    = level-1 [=]
    # level-1 [#]
    == level-2 [==]
    ## level-2 [##]

    Args:
      i: The current character index in self._line.

    Returns:
      -1 if the input line is a heading markdown, i otherwise.
    """
    start_index = i
    marker = self._line[i]
    if marker not in ('=', '#'):
        return start_index
    while i < len(self._line) and self._line[i] == marker:
        i += 1
    if i >= len(self._line) or self._line[i] != ' ':
        return start_index
    if self._line[-1] == marker:
        if not self._line.endswith(self._line[start_index:i]) or self._line[-(i - start_index + 1)] != ' ':
            return start_index
        end_index = -(i - start_index + 1)
    else:
        end_index = len(self._line)
    self._Fill()
    self._buf = self._line[i + 1:end_index]
    heading = self._Attributes()
    if i == 1 and heading.endswith('(1)'):
        self._renderer.SetCommand(heading[:-3].lower().split('_'))
    self._renderer.Heading(i, heading)
    self._depth = 0
    if heading in ['NAME', 'SYNOPSIS']:
        if heading == 'SYNOPSIS':
            is_synopsis_section = True
        else:
            is_synopsis_section = False
        while True:
            self._buf = self._ReadLine()
            if not self._buf:
                break
            self._buf = self._buf.rstrip()
            if self._buf:
                self._renderer.Synopsis(self._Attributes(), is_synopsis=is_synopsis_section)
                break
    elif self._notes and heading == 'NOTES':
        self._buf = self._notes
        self._notes = None
    self._in_example_section = heading == 'EXAMPLES'
    return -1