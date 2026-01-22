from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.core.console import console_attr
from six.moves import range  # pylint: disable=redefined-builtin
def _TransposeListToRows(all_items, width=80, height=40, pad='  ', bold=None, normal=None):
    """Returns padded newline terminated column-wise list for items.

  Used by PromptCompleter to pretty print the possible completions for TAB-TAB.

  Args:
    all_items: [str], The ordered list of all items to transpose.
    width: int, The total display width in characters.
    height: int, The total display height in lines.
    pad: str, String inserted before each column.
    bold: str, The bold font highlight control sequence.
    normal: str, The normal font highlight control sequence.

  Returns:
    [str], A padded newline terminated list of colum-wise rows for the ordered
    items list.  The return value is a single list, not a list of row lists.
    Convert the return value to a printable string by ''.join(return_value).
    The first "row" is preceded by a newline and all rows start with the pad.
  """

    def _Dimensions(items):
        """Returns the transpose dimensions for items."""
        longest_item_len = max((len(x) for x in items))
        column_count = int(width / (len(pad) + longest_item_len)) or 1
        row_count = _IntegerCeilingDivide(len(items), column_count)
        return (longest_item_len, column_count, row_count)

    def _TrimAndAnnotate(item, longest_item_len):
        """Truncates and appends '*' if len(item) > longest_item_len."""
        if len(item) <= longest_item_len:
            return item
        return item[:longest_item_len] + '*'

    def _Highlight(item, longest_item_len, difference_index, bold, normal):
        """Highlights the different part of the completion and left justfies."""
        length = len(item)
        if length > difference_index:
            item = item[:difference_index] + bold + item[difference_index] + normal + item[difference_index + 1:]
        return item + (longest_item_len - length) * ' '
    items = set(all_items)
    longest_item_len, column_count, row_count = _Dimensions(items)
    while row_count > height and longest_item_len > 3:
        items = {_TrimAndAnnotate(x, longest_item_len - 2) for x in all_items}
        longest_item_len, column_count, row_count = _Dimensions(items)
    items = sorted(items)
    if bold:
        difference_index = len(os.path.commonprefix(items))
        items = [_Highlight(x, longest_item_len, difference_index, bold, normal) for x in items]
    row_data = ['\n']
    row_index = 0
    while row_index < row_count:
        column_index = row_index
        for _ in range(column_count):
            if column_index >= len(items):
                break
            row_data.append(pad)
            row_data.append(items[column_index])
            column_index += row_count
        row_data.append('\n')
        row_index += 1
    return row_data