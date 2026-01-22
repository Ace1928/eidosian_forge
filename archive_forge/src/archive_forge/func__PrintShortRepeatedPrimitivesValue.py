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
def _PrintShortRepeatedPrimitivesValue(self, field, value):
    """"Prints short repeated primitives value."""
    self._PrintFieldName(field)
    self.out.write(' [')
    for i in range(len(value) - 1):
        self.PrintFieldValue(field, value[i])
        self.out.write(', ')
    self.PrintFieldValue(field, value[-1])
    self.out.write(']')
    self.out.write(' ' if self.as_one_line else '\n')