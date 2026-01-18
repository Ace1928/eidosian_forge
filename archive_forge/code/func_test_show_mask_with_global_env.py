from unittest import mock
from openstackclient.common import configuration
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
@mock.patch('keystoneauth1.loading.base.get_plugin_options', return_value=opts)
def test_show_mask_with_global_env(self, m_get_plugin_opts):
    arglist = ['--mask']
    verifylist = [('mask', True)]
    self.app.client_manager.configuration_type = 'global_env'
    column_list = ('identity_api_version', 'password', 'region', 'token', 'username')
    datalist = (fakes.VERSION, configuration.REDACTED, fakes.REGION_NAME, configuration.REDACTED, fakes.USERNAME)
    cmd = configuration.ShowConfiguration(self.app, None)
    parsed_args = self.check_parser(cmd, arglist, verifylist)
    columns, data = cmd.take_action(parsed_args)
    self.assertEqual(column_list, columns)
    self.assertEqual(datalist, data)