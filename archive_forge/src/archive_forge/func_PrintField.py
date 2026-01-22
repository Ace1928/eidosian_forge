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
def PrintField(self, field, value):
    """Print a single field name/value pair."""
    self._PrintFieldName(field)
    self.out.write(' ')
    self.PrintFieldValue(field, value)
    self.out.write(' ' if self.as_one_line else '\n')