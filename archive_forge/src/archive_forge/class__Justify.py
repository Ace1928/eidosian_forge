from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import json
import re
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import range  # pylint: disable=redefined-builtin
class _Justify(object):
    """Represents a string object for justification using display width.

  Attributes:
    _adjust: The justification width adjustment. The builtin justification
      functions use len() which counts characters, but some character encodings
      require console_attr.DisplayWidth() which returns the encoded character
      display width.
    _string: The output encoded string to justify.
  """

    def __init__(self, attr, string):
        self._string = console_attr.SafeText(string, encoding=attr.GetEncoding(), escape=False)
        self._adjust = attr.DisplayWidth(self._string) - len(self._string)

    def ljust(self, width):
        return self._string.ljust(width - self._adjust)

    def rjust(self, width):
        return self._string.rjust(width - self._adjust)

    def center(self, width):
        return self._string.center(width - self._adjust)