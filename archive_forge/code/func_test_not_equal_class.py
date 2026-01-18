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
def test_not_equal_class(self):

    class NotRuleDefault(object):

        def __init__(self, name, check_str):
            self.name = name
            self.check = _parser.parse_rule(check_str)
    opt1 = policy.RuleDefault(name='foo', check_str='rule:foo')
    opt2 = NotRuleDefault(name='foo', check_str='rule:foo')
    self.assertNotEqual(opt1, opt2)