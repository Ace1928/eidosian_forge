from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import re
import unicodedata
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import times
import six
class _ExprOperand(object):
    """Operand node.

  Converts an expession value token string to internal string and/or numeric
  values. If an operand has a numeric value then the actual key values are
  converted to numbers at Evaluate() time if possible for Apply(); if the
  conversion fails then the key and operand string values are passed to Apply().

  Attributes:
    list_value: A list of operands.
    numeric_value: The int or float number, or None if the token string does not
      convert to a number.
    string_value: The token string.
  """
    _NUMERIC_CONSTANTS = {'false': 0, 'true': 1}

    def __init__(self, backend, value, normalize=None):
        self.backend = backend
        self.list_value = None
        self.numeric_constant = False
        self.numeric_value = None
        self.string_value = None
        self.Initialize(value, normalize=normalize)

    def Initialize(self, value, normalize=None):
        """Initializes an operand string_value and numeric_value from value.

    Args:
      value: The operand expression string value.
      normalize: Optional normalization function.
    """
        if isinstance(value, list):
            self.list_value = []
            for val in value:
                self.list_value.append(_ExprOperand(self.backend, val, normalize=normalize))
        elif value and normalize:
            self.string_value = normalize(value)
        elif isinstance(value, six.string_types):
            self.string_value = value
            try:
                self.numeric_value = self._NUMERIC_CONSTANTS[value.lower()]
                self.numeric_constant = True
            except KeyError:
                try:
                    self.numeric_value = int(value)
                except ValueError:
                    try:
                        self.numeric_value = float(value)
                    except ValueError:
                        pass
        else:
            self.string_value = _Stringize(value)
            self.numeric_value = value