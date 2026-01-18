import datetime
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from heat.common import crypt
from heat.common import exception
from heat.common import template_format
from heat.db import api as db_api
from heat.engine.clients.os import swift
from heat.engine.clients.os import zaqar
from heat.engine import service
from heat.engine import service_software_config
from heat.engine import software_config_io as swc_io
from heat.objects import resource as resource_objects
from heat.objects import software_config as software_config_object
from heat.objects import software_deployment as software_deployment_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
@mock.patch.object(service_software_config.SoftwareConfigService, 'signal_software_deployment')
@mock.patch.object(swift.SwiftClientPlugin, '_create')
def test_refresh_swift_software_deployment(self, scc, ssd):
    temp_url = 'http://192.0.2.1/v1/AUTH_a/b/c?temp_url_sig=ctemp_url_expires=1234'
    container = 'b'
    object_name = 'c'
    config = self._create_software_config(inputs=[{'name': 'deploy_signal_transport', 'type': 'String', 'value': 'TEMP_URL_SIGNAL'}, {'name': 'deploy_signal_id', 'type': 'String', 'value': temp_url}])
    timeutils.set_time_override(datetime.datetime(2013, 1, 23, 22, 48, 5, 0))
    self.addCleanup(timeutils.clear_time_override)
    now = timeutils.utcnow()
    then = now - datetime.timedelta(0, 60)
    last_modified_1 = 'Wed, 23 Jan 2013 22:47:05 GMT'
    last_modified_2 = 'Wed, 23 Jan 2013 22:48:05 GMT'
    sc = mock.MagicMock()
    headers = {'last-modified': last_modified_1}
    sc.head_object.return_value = headers
    sc.get_object.return_value = (headers, '{"foo": "bar"}')
    scc.return_value = sc
    deployment = self._create_software_deployment(status='IN_PROGRESS', config_id=config['id'])
    deployment_id = str(deployment['id'])
    sd = software_deployment_object.SoftwareDeployment.get_by_id(self.ctx, deployment_id)
    swift_exc = swift.SwiftClientPlugin.exceptions_module
    sc.head_object.side_effect = swift_exc.ClientException('Not found', http_status=404)
    self.assertEqual(sd, self.engine.software_config._refresh_swift_software_deployment(self.ctx, sd, temp_url))
    sc.head_object.assert_called_once_with(container, object_name)
    self.assertEqual([], sc.get_object.mock_calls)
    self.assertEqual([], ssd.mock_calls)
    sc.head_object.side_effect = swift_exc.ClientException('Ouch', http_status=409)
    self.assertRaises(swift_exc.ClientException, self.engine.software_config._refresh_swift_software_deployment, self.ctx, sd, temp_url)
    self.assertEqual([], sc.get_object.mock_calls)
    self.assertEqual([], ssd.mock_calls)
    sc.head_object.side_effect = None
    self.engine.software_config._refresh_swift_software_deployment(self.ctx, sd, temp_url)
    sc.head_object.assert_called_with(container, object_name)
    sc.get_object.assert_called_once_with(container, object_name)
    ssd.assert_called_once_with(self.ctx, deployment_id, {u'foo': u'bar'}, then.isoformat())
    software_deployment_object.SoftwareDeployment.update_by_id(self.ctx, deployment_id, {'updated_at': then})
    sd = software_deployment_object.SoftwareDeployment.get_by_id(self.ctx, deployment_id)
    self.assertEqual(then, sd.updated_at)
    self.engine.software_config._refresh_swift_software_deployment(self.ctx, sd, temp_url)
    sc.get_object.assert_called_once_with(container, object_name)
    ssd.assert_called_once_with(self.ctx, deployment_id, {'foo': 'bar'}, then.isoformat())
    headers['last-modified'] = last_modified_2
    sc.head_object.return_value = headers
    sc.get_object.return_value = (headers, '{"bar": "baz"}')
    self.engine.software_config._refresh_swift_software_deployment(self.ctx, sd, temp_url)
    self.assertEqual(2, len(ssd.mock_calls))
    ssd.assert_called_with(self.ctx, deployment_id, {'bar': 'baz'}, now.isoformat())
    software_deployment_object.SoftwareDeployment.update_by_id(self.ctx, deployment_id, {'updated_at': now})
    sd = software_deployment_object.SoftwareDeployment.get_by_id(self.ctx, deployment_id)
    self.engine.software_config._refresh_swift_software_deployment(self.ctx, sd, temp_url)
    self.assertEqual(2, len(ssd.mock_calls))