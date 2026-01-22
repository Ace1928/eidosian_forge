from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
class ParseCheckTestCase(test_base.BaseTestCase):

    def test_false(self):
        result = _parser._parse_check('!')
        self.assertIsInstance(result, _checks.FalseCheck)

    def test_true(self):
        result = _parser._parse_check('@')
        self.assertIsInstance(result, _checks.TrueCheck)

    @mock.patch.object(_parser, 'LOG')
    def test_bad_rule(self, mock_log):
        result = _parser._parse_check('foobar')
        self.assertIsInstance(result, _checks.FalseCheck)
        mock_log.exception.assert_called_once()

    @mock.patch.object(_checks, 'registered_checks', {})
    @mock.patch.object(_parser, 'LOG')
    def test_no_handler(self, mock_log):
        result = _parser._parse_check('no:handler')
        self.assertIsInstance(result, _checks.FalseCheck)
        mock_log.error.assert_called()

    @mock.patch.object(_checks, 'registered_checks', {'spam': mock.Mock(return_value='spam_check'), None: mock.Mock(return_value='none_check')})
    def test_check(self):
        result = _parser._parse_check('spam:handler')
        self.assertEqual('spam_check', result)
        _checks.registered_checks['spam'].assert_called_once_with('spam', 'handler')
        self.assertFalse(_checks.registered_checks[None].called)

    @mock.patch.object(_checks, 'registered_checks', {None: mock.Mock(return_value='none_check')})
    def test_check_default(self):
        result = _parser._parse_check('spam:handler')
        self.assertEqual('none_check', result)
        _checks.registered_checks[None].assert_called_once_with('spam', 'handler')