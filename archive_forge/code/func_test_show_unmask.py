from unittest import mock
from openstackclient.common import configuration
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
@mock.patch('keystoneauth1.loading.base.get_plugin_options', return_value=opts)
def test_show_unmask(self, m_get_plugin_opts):
    arglist = ['--unmask']
    verifylist = [('mask', False)]
    cmd = configuration.ShowConfiguration(self.app, None)
    parsed_args = self.check_parser(cmd, arglist, verifylist)
    columns, data = cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    datalist = (fakes.PASSWORD, fakes.AUTH_TOKEN, fakes.USERNAME, fakes.VERSION, fakes.REGION_NAME)
    self.assertEqual(datalist, data)