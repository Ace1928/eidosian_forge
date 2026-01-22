from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.console import console_attr as ca
from googlecloudsdk.core.resource import resource_printer_base
import six
class ColumnWidths(object):
    """Computes and stores column widths for a table and any nested tables.

  A nested table is a table defined in the last column of a row in another
  table. ColumnWidths includes any nested tables when computing column widths
  so that the width of each column will be based on the contents of that column
  in the parent table and all nested tables.

  Attributes:
    widths: A list containing the computed minimum width of each column in the
      table and any nested tables.
  """

    def __init__(self, row=None, separator='', skip_empty=False, max_column_width=None, indent_length=0, console_attr=None):
        """Computes the width of each column in row and in any nested tables.

    Args:
      row: An optional list containing the columns in a table row. Any marker
        classes nested within the row must be in the last column of the row.
      separator: An optional separator string to place between columns.
      skip_empty: A boolean indicating whether columns followed only by empty
        columns should be skipped.
      max_column_width: An optional maximum column width.
      indent_length: The number of indent spaces that precede `row`. Added to
        the width of the first column in `row`.
      console_attr: The console attribute for width calculation

    Returns:
      A ColumnWidths object containing the computed column widths.
    """
        self._widths = []
        self._max_column_width = max_column_width
        self._separator_width = len(separator)
        self._skip_empty = skip_empty
        self._indent_length = indent_length
        self._console_attr = console_attr
        if row:
            for i in range(len(row)):
                self._ProcessColumn(i, row)

    @property
    def widths(self):
        """A list containing the minimum width of each column."""
        return self._widths

    def __repr__(self):
        """Returns a string representation of a ColumnWidths object."""
        return '<widths: {}>'.format(self.widths)

    def _SetWidth(self, column_index, content_length):
        """Adjusts widths to account for the length of new column content.

    Args:
      column_index: The column index to potentially update. Must be between 0
        and len(widths).
      content_length: The column content's length to consider when updating
        widths.
    """
        if column_index == len(self._widths):
            self._widths.append(0)
        new_width = max(self._widths[column_index], content_length)
        if self._max_column_width is not None:
            new_width = min(self._max_column_width, new_width)
        self._widths[column_index] = new_width

    def _ProcessColumn(self, index, row):
        """Processes a single column value when computing column widths."""
        record = row[index]
        last_index = len(row) - 1
        if isinstance(record, _Marker):
            if index == last_index:
                self._MergeColumnWidths(record.CalculateColumnWidths(self._max_column_width, self._indent_length + INDENT_STEP))
                return
            else:
                raise TypeError('Markers can only be used in the last column.')
        if _IsLastColumnInRow(row, index, last_index, self._skip_empty):
            self._SetWidth(index, 0)
        else:
            console_attr = self._console_attr
            if self._console_attr is None:
                console_attr = ca.ConsoleAttr()
            width = console_attr.DisplayWidth(str(record)) + self._separator_width
            if index == 0:
                width += self._indent_length
            self._SetWidth(index, width)

    def _MergeColumnWidths(self, other):
        """Merges another ColumnWidths into this instance."""
        for i, width in enumerate(other.widths):
            self._SetWidth(i, width)

    def Merge(self, other):
        """Merges this object and another ColumnWidths into a new ColumnWidths.

    Combines the computed column widths for self and other into a new
    ColumnWidths. Uses the larger maximum column width between the two
    ColumnWidths objects for the merged ColumnWidths. If one or both
    ColumnWidths objects have unlimited max column width (max_column_width is
    None), sets the merged ColumnWidths max column width to unlimited (None).

    Args:
      other: A ColumnWidths object to merge with this instance.

    Returns:
      A new ColumnWidths object containing the combined column widths.
    """
        if not isinstance(other, ColumnWidths):
            raise TypeError('other must be a ColumnWidths object.')
        if self._max_column_width is None or other._max_column_width is None:
            merged_max_column_width = None
        else:
            merged_max_column_width = max(self._max_column_width, other._max_column_width)
        merged = ColumnWidths(max_column_width=merged_max_column_width)
        merged._MergeColumnWidths(self)
        merged._MergeColumnWidths(other)
        return merged