from unittest import mock
from saharaclient.osc import plugin
from saharaclient.tests.unit import base
class TestDataProcessingPlugin(base.BaseTestCase):

    @mock.patch('saharaclient.api.client.Client')
    def test_make_client(self, p_client):
        instance = mock.Mock()
        instance._api_version = {'data_processing': '1.1'}
        instance.session = 'session'
        instance._region_name = 'region_name'
        instance._cli_options.data_processing_url = 'url'
        instance._interface = 'public'
        plugin.make_client(instance)
        p_client.assert_called_with(session='session', region_name='region_name', sahara_url='url', endpoint_type='public')

    @mock.patch('saharaclient.api.client.ClientV2')
    def test_make_client_v2(self, p_client):
        instance = mock.Mock()
        instance._api_version = {'data_processing': '2'}
        instance.session = 'session'
        instance._region_name = 'region_name'
        instance._cli_options.data_processing_url = 'url'
        instance._interface = 'public'
        plugin.make_client(instance)
        p_client.assert_called_with(session='session', region_name='region_name', sahara_url='url', endpoint_type='public')