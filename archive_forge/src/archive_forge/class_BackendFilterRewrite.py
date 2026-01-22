from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.resource import resource_expr_rewrite
class BackendFilterRewrite(resource_expr_rewrite.Backend):
    """Limit filter expressions to those supported by the PrivateCA API backend."""

    def RewriteOperand(self, operand):
        """Always quote the operand as the Cloud Filter Library won't be able to parse as values all arbitrary strings."""
        return self.QuoteOperand(operand, always=True)