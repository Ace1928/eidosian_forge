import logging
import sys
import textwrap
import warnings
import yaml
from oslo_config import cfg
from oslo_serialization import jsonutils
import stevedore
from oslo_policy import policy
def _format_rule_default_yaml(default, include_help=True, comment_rule=True, add_deprecated_rules=True):
    """Create a yaml node from policy.RuleDefault or policy.DocumentedRuleDefault.

    :param default: A policy.RuleDefault or policy.DocumentedRuleDefault object
    :param comment_rule: By default rules will be commented out in generated
                         yaml format text. If you want to keep few or all rules
                         uncommented then pass this arg as False.
    :param add_deprecated_rules: Whether to add the deprecated rules in format
                                 text.
    :returns: A string containing a yaml representation of the RuleDefault
    """
    text = '"%(name)s": "%(check_str)s"\n' % {'name': default.name, 'check_str': default.check_str}
    if include_help:
        op = ''
        if hasattr(default, 'operations'):
            for operation in default.operations:
                if operation['method'] and operation['path']:
                    op += '# %(method)s  %(path)s\n' % {'method': operation['method'], 'path': operation['path']}
        intended_scope = ''
        if getattr(default, 'scope_types', None) is not None:
            intended_scope = '# Intended scope(s): ' + ', '.join(default.scope_types) + '\n'
        comment = '#' if comment_rule else ''
        text = '%(op)s%(scope)s%(comment)s%(text)s\n' % {'op': op, 'scope': intended_scope, 'comment': comment, 'text': text}
        if default.description:
            text = _format_help_text(default.description) + '\n' + text
    if add_deprecated_rules and default.deprecated_for_removal:
        text = '# DEPRECATED\n# "%(name)s" has been deprecated since %(since)s.\n%(reason)s\n%(text)s' % {'name': default.name, 'since': default.deprecated_since, 'reason': _format_help_text(default.deprecated_reason), 'text': text}
    elif add_deprecated_rules and default.deprecated_rule:
        deprecated_reason = default.deprecated_rule.deprecated_reason or default.deprecated_reason
        deprecated_since = default.deprecated_rule.deprecated_since or default.deprecated_since
        deprecated_text = '"%(old_name)s":"%(old_check_str)s" has been deprecated since %(since)s in favor of "%(name)s":"%(check_str)s".' % {'old_name': default.deprecated_rule.name, 'old_check_str': default.deprecated_rule.check_str, 'since': deprecated_since, 'name': default.name, 'check_str': default.check_str}
        text = '%(text)s# DEPRECATED\n%(deprecated_text)s\n%(reason)s\n' % {'text': text, 'reason': _format_help_text(deprecated_reason), 'deprecated_text': _format_help_text(deprecated_text)}
        if default.name != default.deprecated_rule.name:
            text += '# WARNING: A rule name change has been identified.\n#          This may be an artifact of new rules being\n#          included which require legacy fallback\n#          rules to ensure proper policy behavior.\n#          Alternatively, this may just be an alias.\n#          Please evaluate on a case by case basis\n#          keeping in mind the format for aliased\n#          rules is:\n#          "old_rule_name": "new_rule_name".\n'
            text += '# "%(old_name)s": "rule:%(name)s"\n' % {'old_name': default.deprecated_rule.name, 'name': default.name}
        text += '\n'
    return text