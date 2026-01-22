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
def _enforce_scope(self, creds, rule, do_raise=True):
    if creds.get('system'):
        token_scope = 'system'
    elif creds.get('domain_id'):
        token_scope = 'domain'
    else:
        token_scope = 'project'
    result = True
    if token_scope not in rule.scope_types:
        if self.conf.oslo_policy.enforce_scope:
            if do_raise:
                raise InvalidScope(rule, rule.scope_types, token_scope)
            else:
                result = False
        msg = 'Policy %(rule)s failed scope check. The token used to make the request was %(token_scope)s scoped but the policy requires %(policy_scope)s scope. This behavior may change in the future where using the intended scope is required' % {'rule': rule, 'token_scope': token_scope, 'policy_scope': rule.scope_types}
        warnings.warn(msg)
    return result