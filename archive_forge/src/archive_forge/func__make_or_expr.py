import logging
import re
from oslo_policy import _checks
@reducer('check', 'or', 'check')
@reducer('and_expr', 'or', 'check')
def _make_or_expr(self, check1, _or, check2):
    """Create an 'or_expr'.

        Join two checks by the 'or' operator.
        """
    return [('or_expr', _checks.OrCheck([check1, check2]))]