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
def _ParseExpr(self, must=False):
    """Parses an expr term.

    Args:
      must: ExpressionSyntaxError if must is True and there is no expression.

    Raises:
      ExpressionSyntaxError: The expression has a syntax error.

    Returns:
      The new backend expression tree.
    """
    tree = self._ParseAdjTerm()
    if tree:
        tree = self._ParseAdjTail(tree)
    elif must and (not self._backend.IsRewriter()):
        raise resource_exceptions.ExpressionSyntaxError('Term expected [{0}].'.format(self._lex.Annotate()))
    return tree