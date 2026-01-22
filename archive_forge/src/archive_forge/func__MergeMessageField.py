import encodings.raw_unicode_escape  # pylint: disable=unused-import
import encodings.unicode_escape  # pylint: disable=unused-import
import io
import math
import re
from google.protobuf.internal import decoder
from google.protobuf.internal import type_checkers
from google.protobuf import descriptor
from google.protobuf import text_encoding
from google.protobuf import unknown_fields
def _MergeMessageField(self, tokenizer, message, field):
    """Merges a single scalar field into a message.

    Args:
      tokenizer: A tokenizer to parse the field value.
      message: The message of which field is a member.
      field: The descriptor of the field to be merged.

    Raises:
      ParseError: In case of text parsing problems.
    """
    is_map_entry = _IsMapEntry(field)
    if tokenizer.TryConsume('<'):
        end_token = '>'
    else:
        tokenizer.Consume('{')
        end_token = '}'
    if field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
        if field.is_extension:
            sub_message = message.Extensions[field].add()
        elif is_map_entry:
            sub_message = getattr(message, field.name).GetEntryClass()()
        else:
            sub_message = getattr(message, field.name).add()
    else:
        if field.is_extension:
            if not self._allow_multiple_scalars and message.HasExtension(field):
                raise tokenizer.ParseErrorPreviousToken('Message type "%s" should not have multiple "%s" extensions.' % (message.DESCRIPTOR.full_name, field.full_name))
            sub_message = message.Extensions[field]
        else:
            if not self._allow_multiple_scalars and message.HasField(field.name):
                raise tokenizer.ParseErrorPreviousToken('Message type "%s" should not have multiple "%s" fields.' % (message.DESCRIPTOR.full_name, field.name))
            sub_message = getattr(message, field.name)
        sub_message.SetInParent()
    while not tokenizer.TryConsume(end_token):
        if tokenizer.AtEnd():
            raise tokenizer.ParseErrorPreviousToken('Expected "%s".' % (end_token,))
        self._MergeField(tokenizer, sub_message)
    if is_map_entry:
        value_cpptype = field.message_type.fields_by_name['value'].cpp_type
        if value_cpptype == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
            value = getattr(message, field.name)[sub_message.key]
            value.CopyFrom(sub_message.value)
        else:
            getattr(message, field.name)[sub_message.key] = sub_message.value