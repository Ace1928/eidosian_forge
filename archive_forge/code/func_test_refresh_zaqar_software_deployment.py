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
@mock.patch.object(service_software_config.SoftwareConfigService, 'metadata_software_deployments')
@mock.patch.object(db_api, 'resource_update')
@mock.patch.object(db_api, 'resource_get_by_physical_resource_id')
@mock.patch.object(zaqar.ZaqarClientPlugin, 'create_for_tenant')
def test_refresh_zaqar_software_deployment(self, plugin, res_get, res_upd, md_sd, ssd):
    rs = mock.Mock()
    rs.rsrc_metadata = {}
    rs.id = '1234'
    rs.atomic_key = 1
    rd1 = mock.Mock()
    rd1.key = 'user'
    rd1.value = 'user1'
    rd2 = mock.Mock()
    rd2.key = 'password'
    rd2.decrypt_method, rd2.value = crypt.encrypt('pass1')
    rs.data = [rd1, rd2]
    res_get.return_value = rs
    res_upd.return_value = 1
    deployments = {'deploy': 'this'}
    md_sd.return_value = deployments
    config = self._create_software_config(inputs=[{'name': 'deploy_signal_transport', 'type': 'String', 'value': 'ZAQAR_SIGNAL'}, {'name': 'deploy_queue_id', 'type': 'String', 'value': '6789'}])
    queue = mock.Mock()
    zaqar_client = mock.Mock()
    plugin.return_value = zaqar_client
    zaqar_client.queue.return_value = queue
    queue.pop.return_value = [mock.Mock(body='ok')]
    deployment = self._create_software_deployment(status='IN_PROGRESS', config_id=config['id'])
    deployment_id = deployment['id']
    self.assertEqual(deployment, self.engine.show_software_deployment(self.ctx, deployment_id))
    zaqar_client.queue.assert_called_once_with('6789')
    queue.pop.assert_called_once_with()
    ssd.assert_called_once_with(self.ctx, deployment_id, 'ok', None)