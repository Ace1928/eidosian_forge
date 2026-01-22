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
class _ExprRE(_ExprOperator):
    """Unanchored RE match node."""

    def __init__(self, backend, key, operand, transform):
        super(_ExprRE, self).__init__(backend, key, operand, transform)
        self.pattern = _ReCompile(self._operand.string_value)

    def Apply(self, value, unused_operand):
        if not isinstance(value, six.string_types):
            raise TypeError('RE match subject value must be a string.')
        return self.pattern.search(value) is not None