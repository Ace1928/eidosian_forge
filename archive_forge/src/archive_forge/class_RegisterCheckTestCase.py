import os
from unittest import mock
import yaml
import fixtures
from oslo_config import cfg
from oslo_context import context
from oslo_serialization import jsonutils
from oslotest import base as test_base
from oslo_policy import _cache_handler
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy import policy
from oslo_policy.tests import base
class RegisterCheckTestCase(base.PolicyBaseTestCase):

    @mock.patch.object(_checks, 'registered_checks', {})
    def test_register_check(self):

        class TestCheck(policy.Check):
            pass
        policy.register('spam', TestCheck)
        self.assertEqual(dict(spam=TestCheck), _checks.registered_checks)