import logging
import sys
import textwrap
import warnings
import yaml
from oslo_config import cfg
from oslo_serialization import jsonutils
import stevedore
from oslo_policy import policy
def _convert_policy_json_to_yaml(namespace, policy_file, output_file=None):
    with open(policy_file, 'r') as rule_data:
        file_policies = jsonutils.loads(rule_data.read())
    yaml_format_rules = []
    default_policies = get_policies_dict(namespace)
    for section in sorted(default_policies):
        default_rules = default_policies[section]
        for default_rule in default_rules:
            if default_rule.name not in file_policies:
                continue
            file_rule_check_str = file_policies.pop(default_rule.name)
            operations = [{'method': '', 'path': ''}]
            if hasattr(default_rule, 'operations'):
                operations = default_rule.operations
            file_rule = policy.DocumentedRuleDefault(default_rule.name, file_rule_check_str, default_rule.description or default_rule.name, operations, default_rule.deprecated_rule, default_rule.deprecated_for_removal, default_rule.deprecated_reason, default_rule.deprecated_since, scope_types=default_rule.scope_types)
            if file_rule == default_rule:
                rule_text = _format_rule_default_yaml(file_rule, add_deprecated_rules=False)
            else:
                rule_text = _format_rule_default_yaml(file_rule, comment_rule=False, add_deprecated_rules=False)
            yaml_format_rules.append(rule_text)
    extra_rules_text = '# WARNING: Below rules are either deprecated rules\n# or extra rules in policy file, it is strongly\n# recommended to switch to new rules.\n'
    if file_policies:
        yaml_format_rules.append(extra_rules_text)
    for file_rule, check_str in file_policies.items():
        rule_text = '"%(name)s": "%(check_str)s"\n' % {'name': file_rule, 'check_str': check_str}
        yaml_format_rules.append(rule_text)
    if output_file:
        with open(output_file, 'w') as fh:
            fh.writelines(yaml_format_rules)
    else:
        sys.stdout.writelines(yaml_format_rules)