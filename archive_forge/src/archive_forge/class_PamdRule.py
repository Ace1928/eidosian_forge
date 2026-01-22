from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
import os
import re
from tempfile import NamedTemporaryFile
from datetime import datetime
class PamdRule(PamdLine):
    valid_simple_controls = ['required', 'requisite', 'sufficient', 'optional', 'include', 'substack', 'definitive']
    valid_control_values = ['success', 'open_err', 'symbol_err', 'service_err', 'system_err', 'buf_err', 'perm_denied', 'auth_err', 'cred_insufficient', 'authinfo_unavail', 'user_unknown', 'maxtries', 'new_authtok_reqd', 'acct_expired', 'session_err', 'cred_unavail', 'cred_expired', 'cred_err', 'no_module_data', 'conv_err', 'authtok_err', 'authtok_recover_err', 'authtok_lock_busy', 'authtok_disable_aging', 'try_again', 'ignore', 'abort', 'authtok_expired', 'module_unknown', 'bad_item', 'conv_again', 'incomplete', 'default']
    valid_control_actions = ['ignore', 'bad', 'die', 'ok', 'done', 'reset']

    def __init__(self, rule_type, rule_control, rule_path, rule_args=None):
        self.prev = None
        self.next = None
        self._control = None
        self._args = None
        self.rule_type = rule_type
        self.rule_control = rule_control
        self.rule_path = rule_path
        self.rule_args = rule_args

    def matches(self, rule_type, rule_control, rule_path, rule_args=None):
        return rule_type == self.rule_type and rule_control == self.rule_control and (rule_path == self.rule_path)

    @classmethod
    def rule_from_string(cls, line):
        rule_match = RULE_REGEX.search(line)
        rule_args = parse_module_arguments(rule_match.group('args'))
        return cls(rule_match.group('rule_type'), rule_match.group('control'), rule_match.group('path'), rule_args)

    def __str__(self):
        if self.rule_args:
            return '{0: <11}{1} {2} {3}'.format(self.rule_type, self.rule_control, self.rule_path, ' '.join(self.rule_args))
        return '{0: <11}{1} {2}'.format(self.rule_type, self.rule_control, self.rule_path)

    @property
    def rule_control(self):
        if isinstance(self._control, list):
            return '[' + ' '.join(self._control) + ']'
        return self._control

    @rule_control.setter
    def rule_control(self, control):
        if control.startswith('['):
            control = control.replace(' = ', '=').replace('[', '').replace(']', '')
            self._control = control.split(' ')
        else:
            self._control = control

    @property
    def rule_args(self):
        if not self._args:
            return []
        return self._args

    @rule_args.setter
    def rule_args(self, args):
        self._args = parse_module_arguments(args)

    @property
    def line(self):
        return str(self)

    @classmethod
    def is_action_unsigned_int(cls, string_num):
        number = 0
        try:
            number = int(string_num)
        except ValueError:
            return False
        if number >= 0:
            return True
        return False

    @property
    def is_valid(self):
        return self.validate()[0]

    def validate(self):
        if self.rule_type not in VALID_TYPES:
            return (False, 'Rule type, ' + self.rule_type + ', is not valid in rule ' + self.line)
        if isinstance(self._control, str) and self.rule_control not in PamdRule.valid_simple_controls:
            return (False, 'Rule control, ' + self.rule_control + ', is not valid in rule ' + self.line)
        elif isinstance(self._control, list):
            for control in self._control:
                value, action = control.split('=')
                if value not in PamdRule.valid_control_values:
                    return (False, 'Rule control value, ' + value + ', is not valid in rule ' + self.line)
                if action not in PamdRule.valid_control_actions and (not PamdRule.is_action_unsigned_int(action)):
                    return (False, 'Rule control action, ' + action + ', is not valid in rule ' + self.line)
        return (True, 'Rule is valid ' + self.line)