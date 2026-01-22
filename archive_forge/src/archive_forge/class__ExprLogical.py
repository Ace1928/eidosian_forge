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
class _ExprLogical(_Expr):
    """Base logical operator node.

  Attributes:
    left: Left Expr operand.
    right: Right Expr operand.
  """

    def __init__(self, backend, left, right):
        super(_ExprLogical, self).__init__(backend)
        self._left = left
        self._right = right