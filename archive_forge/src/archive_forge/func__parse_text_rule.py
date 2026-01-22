import logging
import re
from oslo_policy import _checks
def _parse_text_rule(rule):
    """Parses policy to the tree.

    Translates a policy written in the policy language into a tree of
    Check objects.
    """
    if not rule:
        return _checks.TrueCheck()
    state = ParseState()
    for tok, value in _parse_tokenize(rule):
        state.shift(tok, value)
    try:
        return state.result
    except ValueError:
        LOG.exception('Failed to understand rule %s', rule)
        return _checks.FalseCheck()