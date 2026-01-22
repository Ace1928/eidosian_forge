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
def ParseFloat(text):
    """Parse a floating point number.

  Args:
    text: Text to parse.

  Returns:
    The number parsed.

  Raises:
    ValueError: If a floating point number couldn't be parsed.
  """
    try:
        return float(text)
    except ValueError:
        if _FLOAT_INFINITY.match(text):
            if text[0] == '-':
                return float('-inf')
            else:
                return float('inf')
        elif _FLOAT_NAN.match(text):
            return float('nan')
        else:
            try:
                return float(text.rstrip('f'))
            except ValueError:
                raise ValueError("Couldn't parse float: %s" % text)