import functools
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def validate_setup_mocks_with_mock(stack, fc, mock_image_constraint=True, validate_create=True):
    instance = stack['WebServer']
    metadata = instance.metadata_get()
    if mock_image_constraint:
        m_image = glance.GlanceClientPlugin.find_image_by_name_or_id
        m_image.assert_called_with(instance.properties['ImageId'])
    user_data = instance.properties['UserData']
    server_userdata = instance.client_plugin().build_userdata(metadata, user_data, 'ec2-user')
    nova.NovaClientPlugin.build_userdata.assert_called_with(metadata, user_data, 'ec2-user')
    if not validate_create:
        return
    fc.servers.create.assert_called_once_with(image=744, flavor=3, key_name='test', name=utils.PhysName(stack.name, 'WebServer'), security_groups=None, userdata=server_userdata, scheduler_hints=None, meta=None, nics=None, availability_zone=None, block_device_mapping=None)