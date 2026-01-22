from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import renderer
def _SplitWideSynopsisGroup(self, group, indent, running_width):
    """Splits a wide SYNOPSIS section group string to self._out.

    Args:
      group: The wide group string to split.
      indent: The prevailing left indent.
      running_width: The width of the self._out line in progress.

    Returns:
      The running_width after the group has been split and written to self._out.
    """
    prev_delimiter = ' '
    while group:
        for delimiter in (' | ', ' : ', ' ', ','):
            part, _, remainder = group.partition(delimiter)
            w = self._attr.DisplayWidth(part)
            if running_width + len(prev_delimiter) + w >= self._width or (prev_delimiter != ',' and delimiter == ','):
                if delimiter != ',' and indent + self.SPLIT_INDENT + len(prev_delimiter) + w >= self._width:
                    continue
                if prev_delimiter == ',':
                    self._out.write(prev_delimiter)
                    prev_delimiter = ' '
                if running_width != indent:
                    running_width = indent + self.SPLIT_INDENT
                    self._out.write('\n' + ' ' * running_width)
            self._out.write(prev_delimiter + part)
            running_width += len(prev_delimiter) + w
            prev_delimiter = delimiter
            group = remainder
            break
    return running_width