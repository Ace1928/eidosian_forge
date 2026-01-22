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
def ConsumeCommentOrTrailingComment(self):
    """Consumes a comment, returns a 2-tuple (trailing bool, comment str)."""
    just_started = self._line == 0 and self._column == 0
    before_parsing = self._previous_line
    comment = self.ConsumeComment()
    trailing = self._previous_line == before_parsing and (not just_started)
    return (trailing, comment)