from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.resource import resource_expr_rewrite
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import times
import six
def _RewriteStrings(self, key, op, operand):
    """Rewrites <key op operand>."""
    terms = []
    for arg in operand if isinstance(operand, list) else [operand]:
        terms.append('{key}{op}{arg}'.format(key=key, op=op, arg=self.Quote(arg, always=True)))
    if len(terms) > 1:
        return '{terms}'.format(terms=' OR '.join(terms))
    return terms[0]