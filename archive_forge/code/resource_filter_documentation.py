from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_expr
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_property
import six
Parses a resource list filter expression.

    This is a hand-rolled recursive descent parser based directly on the
    left-factorized BNF grammar in the file docstring. The parser is not thread
    safe. Each thread should use distinct _Parser objects.

    Args:
      expression: A resource list filter expression string.

    Raises:
      ExpressionSyntaxError: The expression has a syntax error.

    Returns:
      tree: The backend expression tree.
    