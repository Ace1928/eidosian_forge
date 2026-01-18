from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
@mock.patch.object(_parser, '_parse_check', base.FakeCheck)
def test_oneele_zeroele(self):
    result = _parser._parse_list_rule([[]])
    self.assertIsInstance(result, _checks.FalseCheck)
    self.assertEqual('!', str(result))