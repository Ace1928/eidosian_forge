import logging
import sys
import textwrap
import warnings
import yaml
from oslo_config import cfg
from oslo_serialization import jsonutils
import stevedore
from oslo_policy import policy
def _generate_policy(namespace, output_file=None, exclude_deprecated=False):
    """Generate a policy file showing what will be used.

    This takes all registered policies and merges them with what's defined in
    a policy file and outputs the result. That result is the effective policy
    that will be honored by policy checks.

    :param output_file: The path of a file to output to. stdout used if None.
    :param exclude_deprecated: If to exclude deprecated policy rule entries,
                               defaults to False.
    """
    enforcer = _get_enforcer(namespace)
    enforcer.load_rules()
    file_rules = [policy.RuleDefault(name, default.check_str) for name, default in enforcer.file_rules.items()]
    registered_rules = [policy.RuleDefault(name, default.check_str) for name, default in enforcer.registered_rules.items() if name not in enforcer.file_rules]
    policies = {'rules': file_rules + registered_rules}
    output_file = open(output_file, 'w') if output_file else sys.stdout
    for section in _sort_and_format_by_section(policies, include_help=False, exclude_deprecated=exclude_deprecated):
        output_file.write(section)
    if output_file != sys.stdout:
        output_file.close()