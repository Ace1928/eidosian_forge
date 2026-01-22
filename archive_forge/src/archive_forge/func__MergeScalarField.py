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
def _MergeScalarField(self, tokenizer, message, field):
    """Merges a single scalar field into a message.

    Args:
      tokenizer: A tokenizer to parse the field value.
      message: A protocol message to record the data.
      field: The descriptor of the field to be merged.

    Raises:
      ParseError: In case of text parsing problems.
      RuntimeError: On runtime errors.
    """
    _ = self.allow_unknown_extension
    value = None
    if field.type in (descriptor.FieldDescriptor.TYPE_INT32, descriptor.FieldDescriptor.TYPE_SINT32, descriptor.FieldDescriptor.TYPE_SFIXED32):
        value = _ConsumeInt32(tokenizer)
    elif field.type in (descriptor.FieldDescriptor.TYPE_INT64, descriptor.FieldDescriptor.TYPE_SINT64, descriptor.FieldDescriptor.TYPE_SFIXED64):
        value = _ConsumeInt64(tokenizer)
    elif field.type in (descriptor.FieldDescriptor.TYPE_UINT32, descriptor.FieldDescriptor.TYPE_FIXED32):
        value = _ConsumeUint32(tokenizer)
    elif field.type in (descriptor.FieldDescriptor.TYPE_UINT64, descriptor.FieldDescriptor.TYPE_FIXED64):
        value = _ConsumeUint64(tokenizer)
    elif field.type in (descriptor.FieldDescriptor.TYPE_FLOAT, descriptor.FieldDescriptor.TYPE_DOUBLE):
        value = tokenizer.ConsumeFloat()
    elif field.type == descriptor.FieldDescriptor.TYPE_BOOL:
        value = tokenizer.ConsumeBool()
    elif field.type == descriptor.FieldDescriptor.TYPE_STRING:
        value = tokenizer.ConsumeString()
    elif field.type == descriptor.FieldDescriptor.TYPE_BYTES:
        value = tokenizer.ConsumeByteString()
    elif field.type == descriptor.FieldDescriptor.TYPE_ENUM:
        value = tokenizer.ConsumeEnum(field)
    else:
        raise RuntimeError('Unknown field type %d' % field.type)
    if field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
        if field.is_extension:
            message.Extensions[field].append(value)
        else:
            getattr(message, field.name).append(value)
    elif field.is_extension:
        if not self._allow_multiple_scalars and field.has_presence and message.HasExtension(field):
            raise tokenizer.ParseErrorPreviousToken('Message type "%s" should not have multiple "%s" extensions.' % (message.DESCRIPTOR.full_name, field.full_name))
        else:
            message.Extensions[field] = value
    else:
        duplicate_error = False
        if not self._allow_multiple_scalars:
            if field.has_presence:
                duplicate_error = message.HasField(field.name)
            else:
                duplicate_error = bool(getattr(message, field.name))
        if duplicate_error:
            raise tokenizer.ParseErrorPreviousToken('Message type "%s" should not have multiple "%s" fields.' % (message.DESCRIPTOR.full_name, field.name))
        else:
            setattr(message, field.name, value)