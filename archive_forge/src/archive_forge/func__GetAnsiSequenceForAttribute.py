from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core.console.style import mappings
from googlecloudsdk.core.console.style import text
import six
def _GetAnsiSequenceForAttribute(self, text_attributes, style_context):
    """Returns the ANSI start and reset sequences for the text_attributes."""
    style_sequence = ''
    reset_sequence = ''
    attrs = set(getattr(style_context, 'attrs', [])) | set(getattr(text_attributes, 'attrs', []))
    if attrs:
        style_sequence += ';'.join(sorted([six.text_type(attr.value) for attr in attrs]))
        reset_sequence += ';'.join(sorted([six.text_type('%02x' % (attr.value + self.ATTR_OFF)) for attr in attrs]))
    color = getattr(text_attributes, 'color', None) or getattr(style_context, 'color', None)
    if color:
        if style_sequence:
            style_sequence += ';'
        style_sequence += self.SET_FOREGROUND.format(color.value)
        if reset_sequence:
            reset_sequence += ';'
        reset_sequence += self.RESET
    begin_style, end_style = ('', '')
    if style_sequence:
        begin_style = self.CSI + style_sequence + self.SGR
    if reset_sequence:
        end_style = self.CSI + reset_sequence + self.SGR
    return (begin_style, end_style)