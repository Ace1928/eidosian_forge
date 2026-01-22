from openstackclient.common import limits
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
class TestVolumeLimits(volume_fakes.TestVolume):
    absolute_columns = ['Name', 'Value']
    rate_columns = ['Verb', 'URI', 'Value', 'Remain', 'Unit', 'Next Available']

    def setUp(self):
        super().setUp()
        self.app.client_manager.compute_endpoint_enabled = False
        self.fake_limits = volume_fakes.FakeLimits()
        self.volume_client.limits.get.return_value = self.fake_limits

    def test_volume_show_absolute(self):
        arglist = ['--absolute']
        verifylist = [('is_absolute', True)]
        cmd = limits.ShowLimits(self.app, None)
        parsed_args = self.check_parser(cmd, arglist, verifylist)
        columns, data = cmd.take_action(parsed_args)
        ret_limits = list(data)
        compute_reference_limits = self.fake_limits.absolute_limits()
        self.assertEqual(self.absolute_columns, columns)
        self.assertEqual(compute_reference_limits, ret_limits)
        self.assertEqual(10, len(ret_limits))

    def test_volume_show_rate(self):
        arglist = ['--rate']
        verifylist = [('is_rate', True)]
        cmd = limits.ShowLimits(self.app, None)
        parsed_args = self.check_parser(cmd, arglist, verifylist)
        columns, data = cmd.take_action(parsed_args)
        ret_limits = list(data)
        compute_reference_limits = self.fake_limits.rate_limits()
        self.assertEqual(self.rate_columns, columns)
        self.assertEqual(compute_reference_limits, ret_limits)
        self.assertEqual(3, len(ret_limits))