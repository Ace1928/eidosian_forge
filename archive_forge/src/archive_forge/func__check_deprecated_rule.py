import functools
import flask
from oslo_log import log
from oslo_policy import opts
from oslo_policy import policy as common_policy
from oslo_utils import strutils
from keystone.common import authorization
from keystone.common import context
from keystone.common import policies
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
def _check_deprecated_rule(self, action):

    def _name_is_changing(rule):
        deprecated_rule = rule.deprecated_rule
        return deprecated_rule and deprecated_rule.name != rule.name and (deprecated_rule.name in self._enforcer.file_rules)

    def _check_str_is_changing(rule):
        deprecated_rule = rule.deprecated_rule
        return deprecated_rule and deprecated_rule.check_str != rule.check_str and (rule.name not in self._enforcer.file_rules)

    def _is_deprecated_for_removal(rule):
        return rule.deprecated_for_removal and rule.name in self._enforcer.file_rules

    def _emit_warning():
        if not self._enforcer._warning_emitted:
            LOG.warning('Deprecated policy rules found. Use oslopolicy-policy-generator and oslopolicy-policy-upgrade to detect and resolve deprecated policies in your configuration.')
            self._enforcer._warning_emitted = True
    registered_rule = self._enforcer.registered_rules.get(action)
    if not registered_rule:
        return
    if _name_is_changing(registered_rule) or _check_str_is_changing(registered_rule) or _is_deprecated_for_removal(registered_rule):
        _emit_warning()