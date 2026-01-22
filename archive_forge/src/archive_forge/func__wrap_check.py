import logging
import re
from oslo_policy import _checks
@reducer('(', 'check', ')')
@reducer('(', 'and_expr', ')')
@reducer('(', 'or_expr', ')')
def _wrap_check(self, _p1, check, _p2):
    """Turn parenthesized expressions into a 'check' token."""
    return [('check', check)]