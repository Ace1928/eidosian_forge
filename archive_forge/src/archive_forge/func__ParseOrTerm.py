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
def _ParseOrTerm(self, must=False):
    """Parses an orterm term.

    Args:
      must: Raises ExpressionSyntaxError if must is True and there is no
        expression.

    Raises:
      ExpressionSyntaxError: Term expected in expression.

    Returns:
      The new backend expression tree.
    """
    tree = self._ParseAndTerm()
    if tree or self._backend.IsRewriter():
        tree = self._ParseAndTail(tree)
    elif must:
        raise resource_exceptions.ExpressionSyntaxError('Term expected [{0}].'.format(self._lex.Annotate()))
    return tree