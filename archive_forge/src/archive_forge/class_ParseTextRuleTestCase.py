from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
class ParseTextRuleTestCase(test_base.BaseTestCase):

    def test_empty(self):
        result = _parser._parse_text_rule('')
        self.assertIsInstance(result, _checks.TrueCheck)

    @mock.patch.object(_parser, '_parse_tokenize', return_value=[('tok1', 'val1'), ('tok2', 'val2')])
    @mock.patch.object(_parser.ParseState, 'shift')
    @mock.patch.object(_parser.ParseState, 'result', 'result')
    def test_shifts(self, mock_shift, mock_parse_tokenize):
        result = _parser._parse_text_rule('test rule')
        self.assertEqual('result', result)
        mock_parse_tokenize.assert_called_once_with('test rule')
        mock_shift.assert_has_calls([mock.call('tok1', 'val1'), mock.call('tok2', 'val2')])

    @mock.patch.object(_parser, 'LOG', new=mock.Mock())
    @mock.patch.object(_parser, '_parse_tokenize', return_value=[])
    def test_fail(self, mock_parse_tokenize):
        result = _parser._parse_text_rule('test rule')
        self.assertIsInstance(result, _checks.FalseCheck)
        mock_parse_tokenize.assert_called_once_with('test rule')

    def test_A_or_B_or_C(self):
        result = _parser._parse_text_rule('@ or ! or @')
        self.assertEqual('(@ or ! or @)', str(result))

    def test_A_or_B_and_C(self):
        result = _parser._parse_text_rule('@ or ! and @')
        self.assertEqual('(@ or (! and @))', str(result))

    def test_A_and_B_or_C(self):
        result = _parser._parse_text_rule('@ and ! or @')
        self.assertEqual('((@ and !) or @)', str(result))

    def test_A_and_B_and_C(self):
        result = _parser._parse_text_rule('@ and ! and @')
        self.assertEqual('(@ and ! and @)', str(result))

    def test_A_or_B_or_C_or_D(self):
        result = _parser._parse_text_rule('@ or ! or @ or !')
        self.assertEqual('(@ or ! or @ or !)', str(result))

    def test_A_or_B_or_C_and_D(self):
        result = _parser._parse_text_rule('@ or ! or @ and !')
        self.assertEqual('(@ or ! or (@ and !))', str(result))

    def test_A_or_B_and_C_or_D(self):
        result = _parser._parse_text_rule('@ or ! and @ or !')
        self.assertEqual('(@ or (! and @) or !)', str(result))

    def test_A_or_B_and_C_and_D(self):
        result = _parser._parse_text_rule('@ or ! and @ and !')
        self.assertEqual('(@ or (! and @ and !))', str(result))

    def test_A_and_B_or_C_or_D(self):
        result = _parser._parse_text_rule('@ and ! or @ or !')
        self.assertEqual('((@ and !) or @ or !)', str(result))

    def test_A_and_B_or_C_and_D(self):
        result = _parser._parse_text_rule('@ and ! or @ and !')
        self.assertEqual('((@ and !) or (@ and !))', str(result))

    def test_A_and_B_and_C_or_D(self):
        result = _parser._parse_text_rule('@ and ! and @ or !')
        self.assertEqual('((@ and ! and @) or !)', str(result))

    def test_A_and_B_and_C_and_D(self):
        result = _parser._parse_text_rule('@ and ! and @ and !')
        self.assertEqual('(@ and ! and @ and !)', str(result))

    def test_A_and_B_or_C_with_not_1(self):
        result = _parser._parse_text_rule('not @ and ! or @')
        self.assertEqual('((not @ and !) or @)', str(result))

    def test_A_and_B_or_C_with_not_2(self):
        result = _parser._parse_text_rule('@ and not ! or @')
        self.assertEqual('((@ and not !) or @)', str(result))

    def test_A_and_B_or_C_with_not_3(self):
        result = _parser._parse_text_rule('@ and ! or not @')
        self.assertEqual('((@ and !) or not @)', str(result))

    def test_A_and_B_or_C_with_group_1(self):
        for expression in ['( @ ) and ! or @', '@ and ( ! ) or @', '@ and ! or ( @ )', '( @ ) and ! or ( @ )', '@ and ( ! ) or ( @ )', '( @ ) and ( ! ) or ( @ )', '( @ and ! ) or @', '( ( @ ) and ! ) or @', '( @ and ( ! ) ) or @', '( ( @ and ! ) ) or @', '( @ and ! or @ )']:
            result = _parser._parse_text_rule(expression)
            self.assertEqual('((@ and !) or @)', str(result))

    def test_A_and_B_or_C_with_group_2(self):
        result = _parser._parse_text_rule('@ and ( ! or @ )')
        self.assertEqual('(@ and (! or @))', str(result))

    def test_A_and_B_or_C_with_group_and_not_1(self):
        for expression in ['not ( @ ) and ! or @', 'not @ and ( ! ) or @', 'not @ and ! or ( @ )', '( not @ ) and ! or @', '( not @ and ! ) or @', '( not @ and ! or @ )']:
            result = _parser._parse_text_rule(expression)
            self.assertEqual('((not @ and !) or @)', str(result))

    def test_A_and_B_or_C_with_group_and_not_2(self):
        result = _parser._parse_text_rule('not @ and ( ! or @ )')
        self.assertEqual('(not @ and (! or @))', str(result))

    def test_A_and_B_or_C_with_group_and_not_3(self):
        result = _parser._parse_text_rule('not ( @ and ! or @ )')
        self.assertEqual('not ((@ and !) or @)', str(result))

    def test_A_and_B_or_C_with_group_and_not_4(self):
        for expression in ['( @ ) and not ! or @', '@ and ( not ! ) or @', '@ and not ( ! ) or @', '@ and not ! or ( @ )', '( @ and not ! ) or @', '( @ and not ! or @ )']:
            result = _parser._parse_text_rule(expression)
            self.assertEqual('((@ and not !) or @)', str(result))

    def test_A_and_B_or_C_with_group_and_not_5(self):
        result = _parser._parse_text_rule('@ and ( not ! or @ )')
        self.assertEqual('(@ and (not ! or @))', str(result))

    def test_A_and_B_or_C_with_group_and_not_6(self):
        result = _parser._parse_text_rule('@ and not ( ! or @ )')
        self.assertEqual('(@ and not (! or @))', str(result))

    def test_A_and_B_or_C_with_group_and_not_7(self):
        for expression in ['( @ ) and ! or not @', '@ and ( ! ) or not @', '@ and ! or not ( @ )', '@ and ! or ( not @ )', '( @ and ! ) or not @', '( @ and ! or not @ )']:
            result = _parser._parse_text_rule(expression)
            self.assertEqual('((@ and !) or not @)', str(result))

    def test_A_and_B_or_C_with_group_and_not_8(self):
        result = _parser._parse_text_rule('@ and ( ! or not @ )')
        self.assertEqual('(@ and (! or not @))', str(result))