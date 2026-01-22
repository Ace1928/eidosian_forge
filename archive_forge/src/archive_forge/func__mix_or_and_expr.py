import logging
import re
from oslo_policy import _checks
@reducer('or_expr', 'and', 'check')
def _mix_or_and_expr(self, or_expr, _and, check):
    """Modify the case 'A or B and C'"""
    or_expr, check1 = or_expr.pop_check()
    if isinstance(check1, _checks.AndCheck):
        and_expr = check1
        and_expr.add_check(check)
    else:
        and_expr = _checks.AndCheck([check1, check])
    return [('or_expr', or_expr.add_check(and_expr))]