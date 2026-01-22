from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import io
import itertools
import json
import sys
import wcwidth
@staticmethod
def FormatCell(entry, cell_width, cell_height=1, align='c', valign='t'):
    """Format an entry into a list of strings for a fixed cell size.

    Given a (possibly multi-line) entry and a cell height and width,
    we split the entry into a list of lines and format each one into
    the given width and alignment. We then pad the list with
    additional blank lines of the appropriate width so that the
    resulting list has exactly cell_height entries. Each entry
    is also padded with one space on either side.

    We abbreviate strings for width, but we require that the
    number of lines in entry is at most cell_height.

    Args:
      entry: String to format, which may have newlines.
      cell_width: Maximum width for lines in the cell.
      cell_height: Number of lines in the cell.
      align: Alignment to use for lines of text.
      valign: Vertical alignment in the cell. One of 't',
        'c', or 'b' (top, center, and bottom, respectively).

    Returns:
      An iterator yielding exactly cell_height lines, each of
      exact width cell_width + 2, corresponding to this cell.

    Raises:
      FormatterException: If there are too many lines in entry.
      ValueError: If the valign is invalid.
    """
    entry_lines = [PrettyFormatter.Abbreviate(line, cell_width) for line in entry.split('\n')]
    if len(entry_lines) > cell_height:
        raise FormatterException('Too many lines (%s) for a cell of size %s' % (len(entry_lines), cell_height))
    if valign == 't':
        top_lines = []
        bottom_lines = itertools.repeat(' ' * (cell_width + 2), cell_height - len(entry_lines))
    elif valign == 'c':
        top_padding, bottom_padding = PrettyFormatter.CenteredPadding(cell_height, len(entry_lines))
        top_lines = itertools.repeat(' ' * (cell_width + 2), top_padding)
        bottom_lines = itertools.repeat(' ' * (cell_width + 2), bottom_padding)
    elif valign == 'b':
        bottom_lines = []
        top_lines = itertools.repeat(' ' * (cell_width + 2), cell_height - len(entry_lines))
    else:
        raise ValueError('Unknown value for valign: %s' % (valign,))
    content_lines = []
    for line in entry_lines:
        if align == 'c':
            left_padding, right_padding = PrettyFormatter.CenteredPadding(cell_width, wcwidth.wcswidth(line))
            content_lines.append(' %s%s%s ' % (' ' * left_padding, line, ' ' * right_padding))
        elif align in ('l', 'r'):
            padding = ' ' * (cell_width - wcwidth.wcswidth(line))
            fmt = ' %s%s '
            if align == 'l':
                output = fmt % (line, padding)
            else:
                output = fmt % (padding, line)
            content_lines.append(output)
        else:
            raise FormatterException('Unknown alignment: %s' % (align,))
    return itertools.chain(top_lines, content_lines, bottom_lines)