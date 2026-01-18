from collections import abc
from unittest import mock
import hashlib
import os.path
import oslo_config.cfg
from oslo_policy import policy as common_policy
import glance.api.policy
from glance.common import exception
import glance.context
from glance.policies import base as base_policy
from glance.tests.unit import base
def test_policy_file_specified_but_not_found(self):
    """Missing defined policy file should result in a default ruleset"""
    self.config(policy_file='gobble.gobble', group='oslo_policy')
    self.config(enforce_new_defaults=True, group='oslo_policy')
    enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
    context = glance.context.RequestContext(roles=[])
    self.assertRaises(exception.Forbidden, enforcer.enforce, context, 'manage_image_cache', {})
    admin_context = glance.context.RequestContext(roles=['admin'])
    enforcer.enforce(admin_context, 'manage_image_cache', {})