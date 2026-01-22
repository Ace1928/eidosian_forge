from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from urllib import parse as urlparse
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import heat_plugin
from heat.engine.clients.os import swift
from heat.engine.clients.os import zaqar
from heat.engine import environment
from heat.engine.resources.openstack.heat import deployed_server
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _server_create_software_config_zaqar(self, server_name='server_zaqar', md=None):
    stack_name = '%s_s' % server_name
    tmpl, stack = self._setup_test_stack(stack_name, server_zaqar_tmpl)
    self.stack = stack
    props = tmpl.t['resources']['server']['properties']
    props['software_config_transport'] = 'ZAQAR_MESSAGE'
    if md is not None:
        tmpl.t['resources']['server']['metadata'] = md
    self.server_props = props
    resource_defns = tmpl.resource_definitions(stack)
    server = deployed_server.DeployedServer('server', resource_defns['server'], stack)
    zcc = self.patchobject(zaqar.ZaqarClientPlugin, 'create_for_tenant')
    zc = mock.Mock()
    zcc.return_value = zc
    queue = mock.Mock()
    zc.queue.return_value = queue
    scheduler.TaskRunner(server.create)()
    metadata_queue_id = server.data().get('metadata_queue_id')
    md = server.metadata_get()
    queue_id = md['os-collect-config']['zaqar']['queue_id']
    self.assertEqual(queue_id, metadata_queue_id)
    zc.queue.assert_called_once_with(queue_id)
    queue.post.assert_called_once_with({'body': server.metadata_get(), 'ttl': 3600})
    zc.queue.reset_mock()
    server._delete_queue()
    zc.queue.assert_called_once_with(queue_id)
    zc.queue(queue_id).delete.assert_called_once_with()
    return (queue_id, server)