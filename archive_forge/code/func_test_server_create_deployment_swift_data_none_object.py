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
def test_server_create_deployment_swift_data_none_object(self):
    server_name = 'server'
    stack_name = '%s_s' % server_name
    tmpl, stack = self._setup_test_stack(stack_name, ds_deployment_data_none_object_tmpl)
    props = tmpl.t['resources']['server']['properties']
    props['software_config_transport'] = 'POLL_TEMP_URL'
    self.server_props = props
    resource_defns = tmpl.resource_definitions(stack)
    server = deployed_server.DeployedServer(server_name, resource_defns[server_name], stack)
    sc = mock.Mock()
    sc.head_account.return_value = {'x-account-meta-temp-url-key': 'secrit'}
    sc.url = 'http://192.0.2.2'
    self.patchobject(swift.SwiftClientPlugin, '_create', return_value=sc)
    scheduler.TaskRunner(server.create)()
    metadata_put_url = server.data().get('metadata_put_url')
    md = server.metadata_get()
    metadata_url = md['os-collect-config']['request']['metadata_url']
    self.assertNotEqual(metadata_url, metadata_put_url)
    container_name = 'my-custom-container'
    object_name = '0'
    test_path = '/v1/AUTH_test_tenant_id/%s/%s' % (container_name, object_name)
    self.assertEqual(test_path, urlparse.urlparse(metadata_put_url).path)
    self.assertEqual(test_path, urlparse.urlparse(metadata_url).path)
    sc.put_object.assert_called_once_with(container_name, object_name, jsonutils.dumps(md))
    sc.head_container.return_value = {'x-container-object-count': '0'}
    server._delete_temp_url()
    sc.delete_object.assert_called_once_with(container_name, object_name)
    sc.head_container.assert_called_once_with(container_name)
    sc.delete_container.assert_called_once_with(container_name)
    return (metadata_url, server)