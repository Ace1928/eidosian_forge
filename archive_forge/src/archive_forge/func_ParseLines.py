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
def ParseLines(self, lines, message):
    """Parses a text representation of a protocol message into a message."""
    self._allow_multiple_scalars = False
    self._ParseOrMerge(lines, message)
    return message