from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import io
import itertools
import json
import sys
import wcwidth
class PrettyFormatter(TableFormatter):
    """Formats output as an ASCII-art table with borders."""

    def __init__(self, **kwds):
        """Initialize a new PrettyFormatter.

    Keyword arguments:
      junction_char: (default: +) Character to use for table junctions.
      horizontal_char: (default: -) Character to use for horizontal lines.
      vertical_char: (default: |) Character to use for vertical lines.
    """
        super(PrettyFormatter, self).__init__(**kwds)
        self.junction_char = kwds.get('junction_char', '+')
        self.horizontal_char = kwds.get('horizontal_char', '-')
        self.vertical_char = kwds.get('vertical_char', '|')
        self.rows = []
        self.row_heights = []
        self._column_names = []
        self.column_widths = []
        self.column_alignments = []
        self.header_height = 1

    def __len__(self):
        return len(self.rows)

    def __unicode__(self):
        if self or not self.skip_header_when_empty:
            lines = itertools.chain(self.FormatHeader(), self.FormatRows(), self.FormatHrule())
        else:
            lines = []
        return '\n'.join(lines)

    @staticmethod
    def CenteredPadding(interval, size, left_justify=True):
        """Compute information for centering a string in a fixed space.

    Given two integers interval and size, with size <= interval, this
    function computes two integers left_padding and right_padding with
      left_padding + right_padding + size = interval
    and
      |left_padding - right_padding| <= 1.

    In the case that interval and size have different parity,
    left_padding will be larger iff left_justify is True. (That is,
    iff the string should be left justified in the "center" space.)

    Args:
      interval: Size of the fixed space.
      size: Size of the string to center in that space.
      left_justify: (optional, default: True) Whether the string
        should be left-justified in the center space.

    Returns:
      left_padding, right_padding: The size of the left and right
        margins for centering the string.

    Raises:
      FormatterException: If size > interval.
    """
        if size > interval:
            raise FormatterException('Illegal state in table formatting')
        same_parity = interval % 2 == size % 2
        padding = (interval - size) // 2
        if same_parity:
            return (padding, padding)
        elif left_justify:
            return (padding, padding + 1)
        else:
            return (padding + 1, padding)

    @staticmethod
    def Abbreviate(s, width):
        """Abbreviate a string to at most width characters."""
        suffix = '.' * min(width, 3)
        return s if len(s) <= width else s[:width - len(suffix)] + suffix

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

    def FormatRow(self, entries, row_height, column_alignments=None, column_widths=None):
        """Format a row into a list of strings.

    Given a list of entries, which must be the same length as the
    number of columns in this table, and a desired row height, we
    generate a list of strings corresponding to the printed
    representation of that row.

    Args:
      entries: List of entries to format.
      row_height: Number of printed lines corresponding to this row.
      column_alignments: (optional, default self.column_alignments)
        The alignment to use for each column.
      column_widths: (optional, default self.column_widths) The widths
        of each column.

    Returns:
      An iterator over the strings in the printed representation
      of this row.
    """
        column_alignments = column_alignments or self.column_alignments
        column_widths = column_widths or self.column_widths
        curried_format = lambda entry, width, align: self.__class__.FormatCell(str(entry), width, cell_height=row_height, align=align)
        printed_rows = zip(*map(curried_format, entries, column_widths, column_alignments))
        return (self.vertical_char.join(itertools.chain([''], cells, [''])) for cells in printed_rows)

    def HeaderLines(self):
        """Return an iterator over the row(s) for the column names."""
        aligns = itertools.repeat('c')
        return self.FormatRow(self.column_names, self.header_height, column_alignments=aligns)

    def FormatHrule(self):
        """Return a list containing an hrule for this table."""
        entries = (''.join(itertools.repeat('-', width + 2)) for width in self.column_widths)
        return [self.junction_char.join(itertools.chain([''], entries, ['']))]

    def FormatHeader(self):
        """Return an iterator over the lines for the header of this table."""
        return itertools.chain(self.FormatHrule(), self.HeaderLines(), self.FormatHrule())

    def FormatRows(self):
        """Return an iterator over all the rows in this table."""
        return itertools.chain(*map(self.FormatRow, self.rows, self.row_heights))

    def AddRow(self, row):
        """Add a row to this table.

    Args:
      row: A list of length equal to the number of columns in this table.

    Raises:
      FormatterException: If the row length is invalid.
    """
        if len(row) != len(self.column_names):
            raise FormatterException('Invalid row length: %s' % (len(row),))
        split_rows = [str(entry).split('\n') for entry in row]
        self.row_heights.append(max((len(lines) for lines in split_rows)))
        column_widths = (max((wcwidth.wcswidth(line) for line in entry)) for entry in split_rows)
        self.column_widths = [max(width, current) for width, current in zip(column_widths, self.column_widths)]
        self.rows.append(row)

    def AddColumn(self, column_name, align='l', **kwds):
        """Add a column to this table.

    Args:
      column_name: Name for the new column.
      align: (optional, default: 'l') Alignment for the new column entries.

    Raises:
      FormatterException: If the table already has any rows, or if the
        provided alignment is invalid.
    """
        if self:
            raise FormatterException('Cannot add a new column to an initialized table')
        if align not in ('l', 'c', 'r'):
            raise FormatterException('Invalid column alignment: %s' % (align,))
        lines = str(column_name).split('\n')
        self.column_widths.append(max((wcwidth.wcswidth(line) for line in lines)))
        self.column_alignments.append(align)
        self.column_names.append(column_name)
        self.header_height = max(len(lines), self.header_height)

    @property
    def column_names(self):
        return self._column_names