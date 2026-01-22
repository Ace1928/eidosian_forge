import logging
import sys
import textwrap
import warnings
import yaml
from oslo_config import cfg
from oslo_serialization import jsonutils
import stevedore
from oslo_policy import policy
def _list_redundant(namespace):
    """Generate a list of configured policies which match defaults.

    This checks all policies loaded from policy files and checks to see if they
    match registered policies. If so then it is redundant to have them defined
    in a policy file and operators should consider removing them.
    """
    enforcer = _get_enforcer(namespace)
    enforcer.suppress_deprecation_warnings = True
    enforcer.load_rules()
    for name, file_rule in enforcer.file_rules.items():
        reg_rule = enforcer.registered_rules.get(name)
        if reg_rule:
            if file_rule == reg_rule:
                print(reg_rule)