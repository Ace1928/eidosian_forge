from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core.resource import resource_expr_rewrite
class ListRewriter(resource_expr_rewrite.Backend):
    """Project List request resource filter expression rewriter."""

    def Quote(self, value):
        """Returns value double quoted if it contains special characters."""
        return super(ListRewriter, self).Quote(value, always=re.search('[^-@.\\w]', value))

    def RewriteTerm(self, key, op, operand, key_type):
        """Rewrites <key op operand>."""
        del key_type
        if not key.startswith('labels.'):
            key = key.lower()
            if key in ('createtime', 'create_time'):
                key = 'createTime'
            elif key in ('lifecyclestate', 'lifecycle_state'):
                key = 'lifecycleState'
            elif key in ('projectid', 'project_id'):
                key = 'id'
            elif key in ('projectname', 'project_name'):
                key = 'name'
            elif key in ('projectnumber', 'project_number'):
                key = 'projectNumber'
            elif key not in ('id', 'name', 'parent.id', 'parent.type'):
                return None
        if op not in (':', '=', '!='):
            return None
        values = operand if isinstance(operand, list) else [operand]
        parts = []
        for value in values:
            if op == '!=':
                part = 'NOT ({key}{op}{operand})'.format(key=key, op='=', operand=self.Quote(value))
            else:
                part = '{key}{op}{operand}'.format(key=key, op=op, operand=self.Quote(value))
            parts.append(part)
        expr = ' OR '.join(parts)
        if len(parts) > 1:
            expr = '( ' + expr + ' )'
        return expr