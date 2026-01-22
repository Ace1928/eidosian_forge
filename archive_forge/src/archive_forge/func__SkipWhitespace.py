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
def _SkipWhitespace(self):
    while True:
        self._PopLine()
        match = self._whitespace_pattern.match(self._current_line, self._column)
        if not match:
            break
        self.contains_silent_marker_before_current_token = match.group(0) == ' ' + _DEBUG_STRING_SILENT_MARKER
        length = len(match.group(0))
        self._column += length