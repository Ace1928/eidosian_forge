import collections.abc
import copy
import logging
import os
import typing as ty
import warnings
from oslo_config import cfg
from oslo_context import context
from oslo_serialization import jsonutils
from oslo_utils import strutils
import yaml
from oslo_policy import _cache_handler
from oslo_policy import _checks
from oslo_policy._i18n import _
from oslo_policy import _parser
from oslo_policy import opts
def _handle_deprecated_rule(self, default):
    """Handle cases where a policy rule has been deprecated.

        :param default: an instance of RuleDefault that contains an instance of
            DeprecatedRule
        """
    deprecated_rule = default.deprecated_rule
    deprecated_reason = deprecated_rule.deprecated_reason or default.deprecated_reason
    deprecated_since = deprecated_rule.deprecated_since or default.deprecated_since
    deprecated_msg = 'Policy "%(old_name)s":"%(old_check_str)s" was deprecated in %(release)s in favor of "%(name)s":"%(check_str)s". Reason: %(reason)s. Either ensure your deployment is ready for the new default or copy/paste the deprecated policy into your policy file and maintain it manually.' % {'old_name': deprecated_rule.name, 'old_check_str': deprecated_rule.check_str, 'release': deprecated_since, 'name': default.name, 'check_str': default.check_str, 'reason': deprecated_reason}
    if deprecated_rule.name != default.name and deprecated_rule.name in self.file_rules:
        if not self.suppress_deprecation_warnings:
            warnings.warn(deprecated_msg)
        file_rule = self.file_rules[deprecated_rule.name]
        if file_rule.check != deprecated_rule.check and str(file_rule.check) != 'rule:%s' % default.name and (default.name not in self.file_rules.keys()):
            return self.file_rules[deprecated_rule.name].check
    if not self.conf.oslo_policy.enforce_new_defaults and deprecated_rule.check_str != default.check_str and (default.name not in self.file_rules):
        if not (self.suppress_deprecation_warnings or self.suppress_default_change_warnings):
            warnings.warn(deprecated_msg)
        return OrCheck([default.check, deprecated_rule.check])
    return default.check