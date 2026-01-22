import logging
import sys
import textwrap
import warnings
import yaml
from oslo_config import cfg
from oslo_serialization import jsonutils
import stevedore
from oslo_policy import policy
def _validate_policy(namespace):
    """Perform basic sanity checks on a policy file

    Checks for the following errors in the configured policy file:

    * A missing policy file
    * Rules which have invalid syntax
    * Rules which reference non-existent other rules
    * Rules which form a cyclical reference with another rule
    * Rules which do not exist in the specified namespace

    :param namespace: The name under which the oslo.policy enforcer is
                      registered.
    :returns: 0 if all policies validated correctly, 1 if not.
    """
    return_code = 0
    enforcer = _get_enforcer(namespace)
    enforcer.suppress_deprecation_warnings = True
    logging.disable(logging.ERROR)
    enforcer.load_rules()
    if enforcer._informed_no_policy_file:
        print('Configured policy file "%s" not found' % enforcer.policy_file)
        return 1
    logging.disable(logging.NOTSET)
    result = enforcer.check_rules()
    if not result:
        print('Invalid rules found')
        return_code = 1
    with open(cfg.CONF.oslo_policy.policy_file) as f:
        unparsed_policies = yaml.safe_load(f.read())
    for name, file_rule in enforcer.file_rules.items():
        reg_rule = enforcer.registered_rules.get(name)
        if reg_rule is None:
            print('Unknown rule found in policy file:', name)
            return_code = 1
        if str(enforcer.rules[name]) == '!' and unparsed_policies[name] != '!':
            print('Failed to parse rule:', unparsed_policies[name])
            return_code = 1
    return return_code