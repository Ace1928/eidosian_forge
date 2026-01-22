from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import csv
import io
import string
from absl.flags import _helpers
import six
class EnumClassSerializer(ArgumentSerializer):
    """Class for generating string representations of an enum class flag value."""

    def __init__(self, lowercase):
        """Initializes EnumClassSerializer.

    Args:
      lowercase: If True, enum member names are lowercased during serialization.
    """
        self._lowercase = lowercase

    def serialize(self, value):
        """Returns a serialized string of the Enum class value."""
        as_string = _helpers.str_or_unicode(value.name)
        return as_string.lower() if self._lowercase else as_string