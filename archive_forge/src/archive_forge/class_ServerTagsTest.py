from unittest import mock
import uuid
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class ServerTagsTest(common.HeatTestCase):

    def setUp(self):
        super(ServerTagsTest, self).setUp()
        self.fc = fakes_nova.FakeClient()

    def _mock_get_image_id_success(self, imageId_input, imageId):
        self.patchobject(glance.GlanceClientPlugin, 'find_image_by_name_or_id', return_value=imageId)

    def _setup_test_instance(self, intags=None):
        self.stack_name = 'tag_test'
        t = template_format.parse(instance_template)
        template = tmpl.Template(t, env=environment.Environment({'KeyName': 'test'}))
        self.stack = parser.Stack(utils.dummy_context(), self.stack_name, template, stack_id=str(uuid.uuid4()))
        t['Resources']['WebServer']['Properties']['Tags'] = intags
        resource_defns = template.resource_definitions(self.stack)
        instance = instances.Instance('WebServer', resource_defns['WebServer'], self.stack)
        self.patchobject(nova.NovaClientPlugin, 'client', return_value=self.fc)
        self._mock_get_image_id_success('CentOS 5.2', 1)
        self.metadata = instance.metadata_get()
        self.server_userdata = instance.client_plugin().build_userdata(self.metadata, instance.properties['UserData'], 'ec2-user')
        self.mock_build_userdata = self.patchobject(nova.NovaClientPlugin, 'build_userdata', return_value=self.server_userdata)
        self.fc.servers.create = mock.Mock(return_value=self.fc.servers.list()[1])
        return instance

    def test_instance_tags(self):
        tags = [{'Key': 'Food', 'Value': 'yum'}]
        metadata = dict(((tm['Key'], tm['Value']) for tm in tags))
        instance = self._setup_test_instance(intags=tags)
        scheduler.TaskRunner(instance.create)()
        self.mock_build_userdata.assert_called_once_with(self.metadata, instance.properties['UserData'], 'ec2-user')
        self.fc.servers.create.assert_called_once_with(image=1, flavor=1, key_name='test', name=utils.PhysName(self.stack_name, instance.name), security_groups=None, userdata=self.server_userdata, scheduler_hints=None, meta=metadata, nics=None, availability_zone=None, block_device_mapping=None)

    def test_instance_tags_updated(self):
        tags = [{'Key': 'Food', 'Value': 'yum'}]
        metadata = dict(((tm['Key'], tm['Value']) for tm in tags))
        instance = self._setup_test_instance(intags=tags)
        scheduler.TaskRunner(instance.create)()
        self.fc.servers.create.assert_called_once_with(image=1, flavor=1, key_name='test', name=utils.PhysName(self.stack_name, instance.name), security_groups=None, userdata=self.server_userdata, scheduler_hints=None, meta=metadata, nics=None, availability_zone=None, block_device_mapping=None)
        self.assertEqual((instance.CREATE, instance.COMPLETE), instance.state)
        new_tags = [{'Key': 'Food', 'Value': 'yuk'}]
        new_metadata = dict(((tm['Key'], tm['Value']) for tm in new_tags))
        self.fc.servers.set_meta = mock.Mock(return_value=None)
        self.stub_ImageConstraint_validate()
        self.stub_KeypairConstraint_validate()
        self.stub_FlavorConstraint_validate()
        snippet = instance.stack.t.t['Resources'][instance.name]
        props = snippet['Properties'].copy()
        props['Tags'] = new_tags
        update_template = instance.t.freeze(properties=props)
        scheduler.TaskRunner(instance.update, update_template)()
        self.assertEqual((instance.UPDATE, instance.COMPLETE), instance.state)
        self.fc.servers.set_meta.assert_called_once_with(self.fc.servers.list()[1], new_metadata)