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
def test_operation_must_contain_method_and_path_only(self):
    invalid_op = [{'path': '/some/path/', 'method': 'GET', 'break': 'me'}]
    self.assertRaises(policy.InvalidRuleDefault, policy.DocumentedRuleDefault, name='foo', check_str='rule:foo', description='foo_api', operations=invalid_op)