import copy
import json
from heatclient import exc
from oslo_log import log as logging
from testtools import matchers
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
class AutoScalingSignalTest(AutoscalingGroupTest):
    template = '\n{\n  "AWSTemplateFormatVersion" : "2010-09-09",\n  "Description" : "Template to create multiple instances.",\n  "Parameters" : {"size": {"Type": "String", "Default": "1"},\n                  "AZ": {"Type": "String", "Default": "nova"},\n                  "image": {"Type": "String"},\n                  "flavor": {"Type": "String"}},\n  "Resources": {\n    "custom_lb": {\n      "Type": "AWS::EC2::Instance",\n      "Properties": {\n        "ImageId": {"Ref": "image"},\n        "InstanceType": {"Ref": "flavor"},\n        "UserData": "foo",\n        "SecurityGroups": [ "sg-1" ],\n        "Tags": []\n      },\n      "Metadata": {\n        "IPs": {"Fn::GetAtt": ["JobServerGroup", "InstanceList"]}\n      }\n    },\n    "JobServerGroup": {\n      "Type" : "AWS::AutoScaling::AutoScalingGroup",\n      "Properties" : {\n        "AvailabilityZones" : [{"Ref": "AZ"}],\n        "LaunchConfigurationName" : { "Ref" : "JobServerConfig" },\n        "DesiredCapacity" : {"Ref": "size"},\n        "MinSize" : "0",\n        "MaxSize" : "20"\n      }\n    },\n    "JobServerConfig" : {\n      "Type" : "AWS::AutoScaling::LaunchConfiguration",\n      "Metadata": {"foo": "bar"},\n      "Properties": {\n        "ImageId"           : {"Ref": "image"},\n        "InstanceType"      : {"Ref": "flavor"},\n        "SecurityGroups"    : [ "sg-1" ],\n        "UserData"          : "jsconfig data"\n      }\n    },\n    "ScaleUpPolicy" : {\n      "Type" : "AWS::AutoScaling::ScalingPolicy",\n      "Properties" : {\n        "AdjustmentType" : "ChangeInCapacity",\n        "AutoScalingGroupName" : { "Ref" : "JobServerGroup" },\n        "Cooldown" : "0",\n        "ScalingAdjustment": "1"\n      }\n    },\n    "ScaleDownPolicy" : {\n      "Type" : "AWS::AutoScaling::ScalingPolicy",\n      "Properties" : {\n        "AdjustmentType" : "ChangeInCapacity",\n        "AutoScalingGroupName" : { "Ref" : "JobServerGroup" },\n        "Cooldown" : "0",\n        "ScalingAdjustment" : "-2"\n      }\n    }\n  },\n  "Outputs": {\n    "InstanceList": {"Value": {\n      "Fn::GetAtt": ["JobServerGroup", "InstanceList"]}}\n  }\n}\n'
    lb_template = '\nheat_template_version: 2013-05-23\nparameters:\n  ImageId: {type: string}\n  InstanceType: {type: string}\n  SecurityGroups: {type: comma_delimited_list}\n  UserData: {type: string}\n  Tags: {type: comma_delimited_list, default: "x,y"}\n\nresources:\noutputs:\n  PublicIp: {value: "not-used"}\n  AvailabilityZone: {value: \'not-used1\'}\n  PrivateDnsName: {value: \'not-used2\'}\n  PublicDnsName: {value: \'not-used3\'}\n  PrivateIp: {value: \'not-used4\'}\n\n'

    def setUp(self):
        super(AutoScalingSignalTest, self).setUp()
        self.build_timeout = self.conf.build_timeout
        self.build_interval = self.conf.build_interval
        self.files = {'provider.yaml': self.instance_template, 'lb.yaml': self.lb_template}
        self.env = {'resource_registry': {'resources': {'custom_lb': {'AWS::EC2::Instance': 'lb.yaml'}}, 'AWS::EC2::Instance': 'provider.yaml'}, 'parameters': {'size': 2, 'image': self.conf.minimal_image_ref, 'flavor': self.conf.instance_type}}

    def check_instance_count(self, stack_identifier, expected):
        md = self.client.resources.metadata(stack_identifier, 'custom_lb')
        actual_md = len(md['IPs'].split(','))
        if actual_md != expected:
            LOG.warning('check_instance_count exp:%d, meta:%s' % (expected, md['IPs']))
            return False
        stack = self.client.stacks.get(stack_identifier)
        inst_list = self._stack_output(stack, 'InstanceList')
        actual = len(inst_list.split(','))
        if actual != expected:
            LOG.warning('check_instance_count exp:%d, act:%s' % (expected, inst_list))
        return actual == expected

    def test_scaling_meta_update(self):
        """Use heatclient to signal the up and down policy.

        Then confirm that the metadata in the custom_lb is updated each
        time.
        """
        stack_identifier = self.stack_create(template=self.template, files=self.files, environment=self.env)
        self.assertTrue(test.call_until_true(self.build_timeout, self.build_interval, self.check_instance_count, stack_identifier, 2))
        nested_ident = self.assert_resource_is_a_stack(stack_identifier, 'JobServerGroup')
        self.client.resources.signal(stack_identifier, 'ScaleUpPolicy')
        self._wait_for_stack_status(nested_ident, 'UPDATE_COMPLETE')
        self.assertTrue(test.call_until_true(self.build_timeout, self.build_interval, self.check_instance_count, stack_identifier, 3))
        self.client.resources.signal(stack_identifier, 'ScaleDownPolicy')
        self._wait_for_stack_status(nested_ident, 'UPDATE_COMPLETE')
        self.assertTrue(test.call_until_true(self.build_timeout, self.build_interval, self.check_instance_count, stack_identifier, 1))

    def test_signal_with_policy_update(self):
        """Prove that an updated policy is used in the next signal."""
        stack_identifier = self.stack_create(template=self.template, files=self.files, environment=self.env)
        self.assertTrue(test.call_until_true(self.build_timeout, self.build_interval, self.check_instance_count, stack_identifier, 2))
        nested_ident = self.assert_resource_is_a_stack(stack_identifier, 'JobServerGroup')
        self.client.resources.signal(stack_identifier, 'ScaleUpPolicy')
        self._wait_for_stack_status(nested_ident, 'UPDATE_COMPLETE')
        self.assertTrue(test.call_until_true(self.build_timeout, self.build_interval, self.check_instance_count, stack_identifier, 3))
        new_template = self.template.replace('"ScalingAdjustment": "1"', '"ScalingAdjustment": "2"').replace('"DesiredCapacity" : {"Ref": "size"},', '')
        self.update_stack(stack_identifier, template=new_template, environment=self.env, files=self.files)
        self.client.resources.signal(stack_identifier, 'ScaleUpPolicy')
        self._wait_for_stack_status(nested_ident, 'UPDATE_COMPLETE')
        self.assertTrue(test.call_until_true(self.build_timeout, self.build_interval, self.check_instance_count, stack_identifier, 5))

    def test_signal_during_suspend(self):
        """Prove that a signal will fail when the stack is in suspend."""
        stack_identifier = self.stack_create(template=self.template, files=self.files, environment=self.env)
        self.assertTrue(test.call_until_true(self.build_timeout, self.build_interval, self.check_instance_count, stack_identifier, 2))
        nested_ident = self.assert_resource_is_a_stack(stack_identifier, 'JobServerGroup')
        self.client.actions.suspend(stack_id=stack_identifier)
        self._wait_for_stack_status(stack_identifier, 'SUSPEND_COMPLETE')
        ex = self.assertRaises(exc.BadRequest, self.client.resources.signal, stack_identifier, 'ScaleUpPolicy')
        error_msg = 'Signal resource during SUSPEND is not supported'
        self.assertIn(error_msg, str(ex))
        ev = self.wait_for_event_with_reason(stack_identifier, reason='Cannot signal resource during SUSPEND', rsrc_name='ScaleUpPolicy')
        self.assertEqual('SUSPEND_COMPLETE', ev[0].resource_status)
        self._wait_for_stack_status(nested_ident, 'SUSPEND_COMPLETE')
        self._wait_for_stack_status(stack_identifier, 'SUSPEND_COMPLETE')
        self.assertTrue(test.call_until_true(self.build_timeout, self.build_interval, self.check_instance_count, stack_identifier, 2))