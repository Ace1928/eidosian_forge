from unittest import mock
from zunclient import api_versions
from zunclient.osc import plugin
from zunclient.tests.unit import base
class TestContainerPlugin(base.TestCase):

    @mock.patch('zunclient.api_versions.get_api_version')
    @mock.patch('zunclient.v1.client.Client')
    def test_make_client(self, p_client, mock_get_api_version):
        instance = mock.Mock()
        instance._api_version = {'container': '1'}
        instance._region_name = 'zun_region'
        instance.session = 'zun_session'
        mock_get_api_version.return_value = api_versions.APIVersion('1.2')
        plugin.make_client(instance)
        p_client.assert_called_with(region_name='zun_region', session='zun_session', service_type='container', api_version=api_versions.APIVersion('1.2'))