from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
@mock.patch.object(_parser, '_parse_text_rule', return_value='text rule')
@mock.patch.object(_parser, '_parse_list_rule', return_value='list rule')
def test_parse_rule_string(self, mock_parse_list_rule, mock_parse_text_rule):
    result = _parser.parse_rule('a string')
    self.assertEqual('text rule', result)
    self.assertFalse(mock_parse_list_rule.called)
    mock_parse_text_rule.assert_called_once_with('a string')