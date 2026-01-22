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
def _TryCustomFormatMessage(self, message):
    formatted = self.message_formatter(message, self.indent, self.as_one_line)
    if formatted is None:
        return False
    out = self.out
    out.write(' ' * self.indent)
    out.write(formatted)
    out.write(' ' if self.as_one_line else '\n')
    return True