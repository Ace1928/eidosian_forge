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
def _ParseAdjTail(self, tree):
    """Parses an adjtail term.

    Args:
      tree: The backend expression tree.

    Returns:
      The new backend expression tree.
    """
    if not self._lex.IsString('AND', peek=True) and (not self._lex.IsString('OR', peek=True)) and (not self._lex.IsCharacter(')', peek=True)) and (not self._lex.EndOfInput()):
        tree = self._backend.ExprAND(tree, self._ParseExpr(must=True))
    return tree