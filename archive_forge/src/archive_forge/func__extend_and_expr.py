import logging
import re
from oslo_policy import _checks
@reducer('and_expr', 'and', 'check')
def _extend_and_expr(self, and_expr, _and, check):
    """Extend an 'and_expr' by adding one more check."""
    return [('and_expr', and_expr.add_check(check))]