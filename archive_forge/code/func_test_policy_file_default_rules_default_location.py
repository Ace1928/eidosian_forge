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
def test_policy_file_default_rules_default_location(self):
    enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
    context = glance.context.RequestContext(roles=['reader'])
    enforcer.enforce(context, 'get_image', {'project_id': context.project_id})