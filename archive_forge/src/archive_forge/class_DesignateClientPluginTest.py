from unittest import mock
from designateclient import client as designate_client
from heat.common import exception as heat_exception
from heat.engine.clients.os import designate as client
from heat.tests import common
class DesignateClientPluginTest(common.HeatTestCase):

    @mock.patch.object(designate_client, 'Client')
    def test_client(self, client_designate):
        context = mock.Mock()
        session = mock.Mock()
        context.keystone_session = session
        client_plugin = client.DesignateClientPlugin(context)
        self.patchobject(client_plugin, '_get_region_name', return_value='region1')
        client_plugin.client()
        client_designate.assert_called_once_with(endpoint_type='publicURL', service_type='dns', session=session, region_name='region1', version='2')