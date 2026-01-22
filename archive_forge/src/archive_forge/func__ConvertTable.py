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
def _ConvertTable(self, i):
    """Detects and converts a sequence of markdown table lines.

    Markdown attributes are not supported in headings or column data.

    This method will consume multiple input lines if the current line is a
    table heading or separator line. The table markdown sequence is:

      heading line

        heading-1 | ... | heading-n
          OR for boxed table
        | heading-1 | ... | heading-n |

      separator line

        --- | ... | ---
          OR for boxed table
        | --- | ... | --- |
          WHERE
        :---  align left
        :---: align center
        ---:  align right
        ----* length >= fixed_width_length sets column fixed width

      row data lines

        col-1-data-item | ... | col-n-data-item
          ...

      blank line ends table

    Args:
      i: The current character index in self._line.

    Returns:
      -1 if the input lines are table markdown, i otherwise.
    """
    fixed_width_length = 8
    if ' | ' not in self._line:
        return self._ConvertOldTable(i)
    if '---' in self._line:
        head = False
        line = self._line
    else:
        head = True
        line = self._ReadLine()
    if not line or '---' not in line:
        if line is not self._line:
            self._PushBackLine(line)
        return self._ConvertOldTable(i)
    box = False
    if head:
        heading = re.split(' *\\| *', self._line.strip())
        if not heading[0] and (not heading[-1]):
            heading = heading[1:-1]
            box = True
    else:
        heading = []
    sep = re.split(' *\\| *', line.strip())
    if not sep[0] and (not sep[-1]):
        sep = sep[1:-1]
        box = True
    if heading and len(heading) != len(sep):
        if line is not self._line:
            self._PushBackLine(line)
        return self._ConvertOldTable(i)
    table = renderer.TableAttributes(box=box)
    for index in range(len(sep)):
        align = 'left'
        s = sep[index]
        if s.startswith(':'):
            if s.endswith(':'):
                align = 'center'
        elif s.endswith(':'):
            align = 'right'
        label = heading[index] if index < len(heading) else None
        width = len(s) if len(s) >= fixed_width_length else 0
        table.AddColumn(align=align, label=label, width=width)
    rows = []
    while True:
        line = self._ReadLine()
        if line in (None, '', '\n', '+\n'):
            self._PushBackLine(line)
            break
        row = re.split(' *\\| *', line.rstrip())
        rows.append(row)
    if rows:
        self._renderer.Table(table, rows)
    self._buf = ''
    return -1