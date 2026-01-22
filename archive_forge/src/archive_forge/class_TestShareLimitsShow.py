import ddt
from manilaclient import api_versions
from manilaclient.api_versions import MAX_VERSION
from manilaclient.osc.v2 import share_limits as osc_share_limits
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.ddt
class TestShareLimitsShow(TestShareLimits):

    def setUp(self):
        super(TestShareLimitsShow, self).setUp()
        self.cmd = osc_share_limits.ShareLimitsShow(self.app, None)
        self.absolute_limit_columns = ['Name', 'Value']
        self.rate_limit_columns = ['Verb', 'Regex', 'URI', 'Value', 'Remaining', 'Unit', 'Next Available']

    @ddt.data('absolute', 'rate')
    def test_limits(self, limit_type):
        share_limits = manila_fakes.FakeShareLimits.create_one_share_limit()
        self.share_limits_mock.get.return_value = share_limits
        expected_data = share_limits._info[f'{limit_type}_limit']
        arglist = []
        verifylist = []
        if limit_type == 'absolute':
            arglist.append('--absolute')
            verifylist.extend([('absolute', True), ('rate', False)])
        else:
            arglist.append('--rate')
            verifylist.extend([('absolute', False), ('rate', True)])
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        actual_columns, actual_data = self.cmd.take_action(parsed_args)
        self.assertEqual(getattr(self, f'{limit_type}_limit_columns'), actual_columns)
        if limit_type == 'rate':
            expected_data_tuple = tuple(expected_data.values())
            self.assertEqual(sorted(expected_data_tuple), sorted(next(actual_data)))
        else:
            expected_data_tuple = tuple(expected_data.items())
            self.assertEqual(sorted(expected_data_tuple), sorted(actual_data))