from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import json
import re
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _GetNextLineAndRemainder(self, s, max_width, include_all_whitespace=False):
    """Helper function to get next line of wrappable text."""
    current_width = 0
    split = 0
    prefix = ''
    while split < len(s):
        if self._csi and s[split:].startswith(self._csi):
            seq_length = self._console_attr.GetControlSequenceLen(s[split:])
            prefix = s[split:split + seq_length]
            split += seq_length
        else:
            current_width += console_attr.GetCharacterDisplayWidth(s[split])
            if current_width > max_width:
                break
            split += 1
    if not include_all_whitespace:
        split += len(s[split:]) - len(s[split:].lstrip())
    first_newline = re.search('\\n', s)
    if first_newline and first_newline.end() <= split:
        split = first_newline.end()
    else:
        max_whitespace = None
        for r in re.finditer('\\s+', s):
            if r.end() > split:
                if include_all_whitespace and r.start() <= split:
                    max_whitespace = split
                break
            max_whitespace = r.end()
        if max_whitespace:
            split = max_whitespace
    if not include_all_whitespace:
        next_line = s[:split].rstrip()
    else:
        next_line = s[:split]
    remaining_value = s[split:]
    if prefix and prefix != self._console_attr.GetFontCode():
        next_line += self._console_attr.GetFontCode()
        remaining_value = prefix + remaining_value
    return (next_line, remaining_value)