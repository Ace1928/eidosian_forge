from __future__ import absolute_import, division, print_function
from . import utils
def _do_rules_differ(current_rules, desired_rules):
    if len(current_rules) != len(desired_rules):
        return True
    if _rule_set(current_rules) != _rule_set(desired_rules):
        return True
    return False