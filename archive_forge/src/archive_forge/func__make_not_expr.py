import logging
import re
from oslo_policy import _checks
@reducer('not', 'check')
def _make_not_expr(self, _not, check):
    """Invert the result of another check."""
    return [('check', _checks.NotCheck(check))]