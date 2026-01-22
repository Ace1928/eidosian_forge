from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import platform
import re
import subprocess
import sys
from googlecloudsdk.core.util import encoding
class InvalidEnumValue(Error):
    """Exception for when a string could not be parsed to a valid enum value."""

    def __init__(self, given, enum_type, options):
        """Constructs a new exception.

    Args:
      given: str, The given string that could not be parsed.
      enum_type: str, The human readable name of the enum you were trying to
        parse.
      options: list(str), The valid values for this enum.
    """
        super(InvalidEnumValue, self).__init__('Could not parse [{0}] into a valid {1}.  Valid values are [{2}]'.format(given, enum_type, ', '.join(options)))