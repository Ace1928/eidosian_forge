from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import csv
import io
import string
from absl.flags import _helpers
import six
class EnumParser(ArgumentParser):
    """Parser of a string enum value (a string value from a given set)."""

    def __init__(self, enum_values, case_sensitive=True):
        """Initializes EnumParser.

    Args:
      enum_values: [str], a non-empty list of string values in the enum.
      case_sensitive: bool, whether or not the enum is to be case-sensitive.

    Raises:
      ValueError: When enum_values is empty.
    """
        if not enum_values:
            raise ValueError('enum_values cannot be empty, found "{}"'.format(enum_values))
        super(EnumParser, self).__init__()
        self.enum_values = enum_values
        self.case_sensitive = case_sensitive

    def parse(self, argument):
        """Determines validity of argument and returns the correct element of enum.

    Args:
      argument: str, the supplied flag value.

    Returns:
      The first matching element from enum_values.

    Raises:
      ValueError: Raised when argument didn't match anything in enum.
    """
        if self.case_sensitive:
            if argument not in self.enum_values:
                raise ValueError('value should be one of <%s>' % '|'.join(self.enum_values))
            else:
                return argument
        elif argument.upper() not in [value.upper() for value in self.enum_values]:
            raise ValueError('value should be one of <%s>' % '|'.join(self.enum_values))
        else:
            return [value for value in self.enum_values if value.upper() == argument.upper()][0]

    def flag_type(self):
        """See base class."""
        return 'string enum'