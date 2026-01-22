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
def _TryPrintAsAnyMessage(self, message):
    """Serializes if message is a google.protobuf.Any field."""
    if '/' not in message.type_url:
        return False
    packed_message = _BuildMessageFromTypeName(message.TypeName(), self.descriptor_pool)
    if packed_message:
        packed_message.MergeFromString(message.value)
        colon = ':' if self.force_colon else ''
        self.out.write('%s[%s]%s ' % (self.indent * ' ', message.type_url, colon))
        self._PrintMessageFieldValue(packed_message)
        self.out.write(' ' if self.as_one_line else '\n')
        return True
    else:
        return False