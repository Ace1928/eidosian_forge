import datetime
import json
from unittest import mock
from oslo_utils import timeutils
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class TestGroupCrud(common.HeatTestCase):

    def setUp(self):
        super(TestGroupCrud, self).setUp()
        t = template_format.parse(as_template)
        self.stack = utils.parse_stack(t, params=inline_templates.as_params)
        self.group = self.stack['WebServerGroup']
        self.assertIsNone(self.group.validate())

    def test_handle_create(self):
        self.group.create_with_template = mock.Mock(return_value=None)
        self.group.child_template = mock.Mock(return_value='{}')
        self.group.handle_create()
        self.group.child_template.assert_called_once_with()
        self.group.create_with_template.assert_called_once_with('{}')

    def test_handle_update_desired_cap(self):
        self.group._try_rolling_update = mock.Mock(return_value=None)
        self.group.resize = mock.Mock(return_value=None)
        props = {'DesiredCapacity': 4, 'MinSize': 0, 'MaxSize': 6}
        self.group._get_new_capacity = mock.Mock(return_value=4)
        defn = rsrc_defn.ResourceDefinition('nopayload', 'AWS::AutoScaling::AutoScalingGroup', props)
        self.group.handle_update(defn, None, props)
        self.group.resize.assert_called_once_with(4)
        self.group._try_rolling_update.assert_called_once_with(props)

    def test_handle_update_desired_nocap(self):
        self.group._try_rolling_update = mock.Mock(return_value=None)
        self.group.resize = mock.Mock(return_value=None)
        get_size = self.patchobject(grouputils, 'get_size')
        get_size.return_value = 4
        props = {'MinSize': 0, 'MaxSize': 6}
        defn = rsrc_defn.ResourceDefinition('nopayload', 'AWS::AutoScaling::AutoScalingGroup', props)
        self.group.handle_update(defn, None, props)
        self.group.resize.assert_called_once_with(4)
        self.group._try_rolling_update.assert_called_once_with(props)

    def test_conf_properties_vpc_zone(self):
        self.stub_ImageConstraint_validate()
        self.stub_FlavorConstraint_validate()
        self.stub_SnapshotConstraint_validate()
        t = template_format.parse(as_template)
        properties = t['Resources']['WebServerGroup']['Properties']
        properties['VPCZoneIdentifier'] = ['xxxx']
        stack = utils.parse_stack(t, params=inline_templates.as_params)
        conf = stack['LaunchConfig']
        self.assertIsNone(conf.validate())
        scheduler.TaskRunner(conf.create)()
        self.assertEqual((conf.CREATE, conf.COMPLETE), conf.state)
        group = stack['WebServerGroup']
        config, props = group._get_conf_properties()
        self.assertEqual('xxxx', props['SubnetId'])
        conf.delete()

    def test_update_in_failed(self):
        self.group.state_set('CREATE', 'FAILED')
        self.group.resize = mock.Mock(return_value=None)
        new_defn = rsrc_defn.ResourceDefinition('asg', 'AWS::AutoScaling::AutoScalingGroup', {'AvailabilityZones': ['nova'], 'LaunchConfigurationName': 'config', 'MaxSize': 5, 'MinSize': 1, 'DesiredCapacity': 2})
        self.group.handle_update(new_defn, None, None)
        self.group.resize.assert_called_once_with(2)