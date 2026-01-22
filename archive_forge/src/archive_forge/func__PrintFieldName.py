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
def _PrintFieldName(self, field):
    """Print field name."""
    out = self.out
    out.write(' ' * self.indent)
    if self.use_field_number:
        out.write(str(field.number))
    elif field.is_extension:
        out.write('[')
        if field.containing_type.GetOptions().message_set_wire_format and field.type == descriptor.FieldDescriptor.TYPE_MESSAGE and (field.label == descriptor.FieldDescriptor.LABEL_OPTIONAL):
            out.write(field.message_type.full_name)
        else:
            out.write(field.full_name)
        out.write(']')
    elif field.type == descriptor.FieldDescriptor.TYPE_GROUP:
        out.write(field.message_type.name)
    else:
        out.write(field.name)
    if self.force_colon or field.cpp_type != descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
        out.write(':')