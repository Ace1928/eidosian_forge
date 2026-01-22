from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.resource import resource_expr_rewrite
def RewriteTerm(self, key, op, operand, key_type):
    """Rewrites <key op operand>."""
    del key_type
    if op != ':':
        raise OperatorNotSupportedError('The [{}] operator is not supported.'.format(op))
    return [{operand: key}]