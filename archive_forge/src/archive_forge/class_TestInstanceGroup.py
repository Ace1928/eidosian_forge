import copy
from unittest import mock
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.heat import instance_group as instgrp
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
class TestInstanceGroup(common.HeatTestCase):

    def setUp(self):
        super(TestInstanceGroup, self).setUp()
        t = template_format.parse(inline_templates.as_template)
        self.stack = utils.parse_stack(t, params=inline_templates.as_params)
        self.defn = rsrc_defn.ResourceDefinition('asg', 'OS::Heat::InstanceGroup', {'Size': 2, 'AvailabilityZones': ['zoneb'], 'LaunchConfigurationName': 'config'})
        self.instance_group = instgrp.InstanceGroup('asg', self.defn, self.stack)

    def test_update_timeout(self):
        self.stack.timeout_secs = mock.Mock(return_value=100)
        self.assertEqual(60, self.instance_group._update_timeout(batch_cnt=3, pause_sec=20))

    def test_child_template(self):
        self.instance_group._create_template = mock.Mock(return_value='tpl')
        self.assertEqual('tpl', self.instance_group.child_template())
        self.instance_group._create_template.assert_called_once_with(2)

    def test_child_params(self):
        expected = {'parameters': {}, 'resource_registry': {'OS::Heat::ScaledResource': 'AWS::EC2::Instance'}}
        self.assertEqual(expected, self.instance_group.child_params())

    def test_tags_default(self):
        expected = [{'Value': u'asg', 'Key': 'metering.groupname'}]
        self.assertEqual(expected, self.instance_group._tags())

    def test_tags_with_extra(self):
        self.instance_group.properties.data['Tags'] = [{'Key': 'fee', 'Value': 'foo'}]
        expected = [{'Key': 'fee', 'Value': 'foo'}, {'Value': u'asg', 'Key': 'metering.groupname'}]
        self.assertEqual(expected, self.instance_group._tags())

    def test_tags_with_metering(self):
        self.instance_group.properties.data['Tags'] = [{'Key': 'metering.fee', 'Value': 'foo'}]
        expected = [{'Key': 'metering.fee', 'Value': 'foo'}]
        self.assertEqual(expected, self.instance_group._tags())

    def test_validate_launch_conf_ref(self):
        props = self.instance_group.properties.data
        props['LaunchConfigurationName'] = 'JobServerConfig'
        error = self.assertRaises(ValueError, self.instance_group.validate)
        self.assertIn('(JobServerConfig) reference can not be found', str(error))
        props['LaunchConfigurationName'] = 'LaunchConfig'
        error = self.assertRaises(ValueError, self.instance_group.validate)
        self.assertIn('LaunchConfigurationName (LaunchConfig) requires a reference to the configuration not just the name of the resource.', str(error))
        self.instance_group.name = 'WebServerGroup'
        self.instance_group.validate()

    def test_handle_create(self):
        self.instance_group.create_with_template = mock.Mock(return_value=None)
        self.instance_group._create_template = mock.Mock(return_value='{}')
        self.instance_group.handle_create()
        self.instance_group._create_template.assert_called_once_with(2)
        self.instance_group.create_with_template.assert_called_once_with('{}')

    def test_update_in_failed(self):
        self.instance_group.state_set('CREATE', 'FAILED')
        self.instance_group.resize = mock.Mock(return_value=None)
        self.instance_group.handle_update(self.defn, None, None)
        self.instance_group.resize.assert_called_once_with(2)

    def test_handle_delete(self):
        self.instance_group.delete_nested = mock.Mock(return_value=None)
        self.instance_group.handle_delete()
        self.instance_group.delete_nested.assert_called_once_with()

    def test_handle_update_size(self):
        self.instance_group._try_rolling_update = mock.Mock(return_value=None)
        self.instance_group.resize = mock.Mock(return_value=None)
        props = {'Size': 5}
        defn = rsrc_defn.ResourceDefinition('nopayload', 'AWS::AutoScaling::AutoScalingGroup', props)
        self.instance_group.handle_update(defn, None, props)
        self.instance_group.resize.assert_called_once_with(5)

    def test_attributes(self):
        get_output = mock.Mock(return_value={'z': '2.1.3.1', 'x': '2.1.3.2', 'c': '2.1.3.3'})
        self.instance_group.get_output = get_output
        inspector = self.instance_group._group_data()
        inspector.member_names = mock.Mock(return_value=['z', 'x', 'c'])
        res = self.instance_group._resolve_attribute('InstanceList')
        self.assertEqual('2.1.3.1,2.1.3.2,2.1.3.3', res)
        get_output.assert_called_once_with('InstanceList')

    def test_attributes_format_fallback(self):
        self.instance_group.get_output = mock.Mock(return_value=['2.1.3.2', '2.1.3.1', '2.1.3.3'])
        mock_members = self.patchobject(grouputils, 'get_members')
        instances = []
        for ip_ex in range(1, 4):
            inst = mock.Mock()
            inst.FnGetAtt.return_value = '2.1.3.%d' % ip_ex
            instances.append(inst)
        mock_members.return_value = instances
        res = self.instance_group._resolve_attribute('InstanceList')
        self.assertEqual('2.1.3.1,2.1.3.2,2.1.3.3', res)

    def test_attributes_fallback(self):
        self.instance_group.get_output = mock.Mock(side_effect=exception.NotFound)
        mock_members = self.patchobject(grouputils, 'get_members')
        instances = []
        for ip_ex in range(1, 4):
            inst = mock.Mock()
            inst.FnGetAtt.return_value = '2.1.3.%d' % ip_ex
            instances.append(inst)
        mock_members.return_value = instances
        res = self.instance_group._resolve_attribute('InstanceList')
        self.assertEqual('2.1.3.1,2.1.3.2,2.1.3.3', res)

    def test_instance_group_refid_rsrc_name(self):
        self.instance_group.id = '123'
        self.instance_group.uuid = '9bfb9456-3fe8-41f4-b318-9dba18eeef74'
        self.instance_group.action = 'CREATE'
        expected = self.instance_group.name
        self.assertEqual(expected, self.instance_group.FnGetRefId())

    def test_instance_group_refid_rsrc_id(self):
        self.instance_group.resource_id = 'phy-rsrc-id'
        self.assertEqual('phy-rsrc-id', self.instance_group.FnGetRefId())