from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.core.console import console_attr
from six.moves import range  # pylint: disable=redefined-builtin
def _Highlight(item, longest_item_len, difference_index, bold, normal):
    """Highlights the different part of the completion and left justfies."""
    length = len(item)
    if length > difference_index:
        item = item[:difference_index] + bold + item[difference_index] + normal + item[difference_index + 1:]
    return item + (longest_item_len - length) * ' '