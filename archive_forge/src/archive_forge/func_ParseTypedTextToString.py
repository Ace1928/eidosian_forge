from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core.console.style import mappings
from googlecloudsdk.core.console.style import text
import six
def ParseTypedTextToString(self, typed_text, style_context=None, stylize=True):
    """Parses a TypedText object into plain and ansi-annotated unicode.

    The reason this returns both the plain and ansi-annotated strings is to
    support file logging.

    Args:
      typed_text: mappings.TypedText, typed text to be converted to unicode.
      style_context: _StyleContext, argument used for recursive calls
        to preserve text attributes and colors. Recursive calls are made when a
        TypedText object contains TypedText objects.
      stylize: bool, Whether or not to stylize the string.

    Returns:
      str, the parsed text.
    """
    if isinstance(typed_text, six.string_types):
        return typed_text
    stylize = stylize and self.style_enabled
    parsed_chunks = []
    text_attributes = self.style_mappings[typed_text.text_type]
    begin_style, end_style = self._GetAnsiSequenceForAttribute(text_attributes, style_context)
    if style_context:
        new_style_context = style_context.UpdateFromTextAttributes(text_attributes)
    else:
        new_style_context = _StyleContext.FromTextAttributes(text_attributes)
    for chunk in typed_text.texts:
        if isinstance(chunk, text.TypedText):
            parsed_chunks.append(self.ParseTypedTextToString(chunk, style_context=new_style_context, stylize=stylize))
            if stylize:
                parsed_chunks.append(begin_style)
        else:
            parsed_chunks.append(chunk)
    parsed_text = ''.join(parsed_chunks)
    if text_attributes and text_attributes.format_str:
        parsed_text = text_attributes.format_str.format(parsed_text)
    if stylize:
        parsed_text = '{begin_style}{text}{end_style}'.format(begin_style=begin_style, text=parsed_text, end_style=end_style)
    return parsed_text