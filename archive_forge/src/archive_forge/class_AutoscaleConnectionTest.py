import time
from boto.ec2.autoscale import AutoScaleConnection
from boto.ec2.autoscale.activity import Activity
from boto.ec2.autoscale.group import AutoScalingGroup, ProcessType
from boto.ec2.autoscale.launchconfig import LaunchConfiguration
from boto.ec2.autoscale.policy import AdjustmentType, MetricCollectionTypes, ScalingPolicy
from boto.ec2.autoscale.scheduled import ScheduledUpdateGroupAction
from boto.ec2.autoscale.instance import Instance
from boto.ec2.autoscale.tag import Tag
from tests.compat import unittest
class AutoscaleConnectionTest(unittest.TestCase):
    ec2 = True
    autoscale = True

    def test_basic(self):
        print('--- running %s tests ---' % self.__class__.__name__)
        c = AutoScaleConnection()
        self.assertTrue(repr(c).startswith('AutoScaleConnection'))
        groups = c.get_all_groups()
        for group in groups:
            self.assertIsInstance(group, AutoScalingGroup)
            activities = group.get_activities()
            for activity in activities:
                self.assertIsInstance(activity, Activity)
        configs = c.get_all_launch_configurations()
        for config in configs:
            self.assertIsInstance(config, LaunchConfiguration)
        policies = c.get_all_policies()
        for policy in policies:
            self.assertIsInstance(policy, ScalingPolicy)
        actions = c.get_all_scheduled_actions()
        for action in actions:
            self.assertIsInstance(action, ScheduledUpdateGroupAction)
        instances = c.get_all_autoscaling_instances()
        for instance in instances:
            self.assertIsInstance(instance, Instance)
        ptypes = c.get_all_scaling_process_types()
        for ptype in ptypes:
            self.assertTrue(ptype, ProcessType)
        adjustments = c.get_all_adjustment_types()
        for adjustment in adjustments:
            self.assertIsInstance(adjustment, AdjustmentType)
        types = c.get_all_metric_collection_types()
        self.assertIsInstance(types, MetricCollectionTypes)
        time_string = '%d' % int(time.time())
        lc_name = 'lc-%s' % time_string
        lc = LaunchConfiguration(name=lc_name, image_id='ami-2272864b', instance_type='t1.micro')
        c.create_launch_configuration(lc)
        found = False
        lcs = c.get_all_launch_configurations()
        for lc in lcs:
            if lc.name == lc_name:
                found = True
                break
        assert found
        group_name = 'group-%s' % time_string
        group = AutoScalingGroup(name=group_name, launch_config=lc, availability_zones=['us-east-1a'], min_size=1, max_size=1)
        c.create_auto_scaling_group(group)
        found = False
        groups = c.get_all_groups()
        for group in groups:
            if group.name == group_name:
                found = True
                break
        assert found
        tag = Tag(key='foo', value='bar', resource_id=group_name, propagate_at_launch=True)
        c.create_or_update_tags([tag])
        found = False
        tags = c.get_all_tags()
        for tag in tags:
            if tag.resource_id == group_name and tag.key == 'foo':
                found = True
                break
        assert found
        c.delete_tags([tag])
        group.shutdown_instances()
        instances = True
        while instances:
            time.sleep(5)
            groups = c.get_all_groups()
            for group in groups:
                if group.name == group_name:
                    if not group.instances:
                        instances = False
        group.delete()
        lc.delete()
        found = True
        while found:
            found = False
            time.sleep(5)
            tags = c.get_all_tags()
            for tag in tags:
                if tag.resource_id == group_name and tag.key == 'foo':
                    found = True
        assert not found
        print('--- tests completed ---')

    def test_ebs_optimized_regression(self):
        c = AutoScaleConnection()
        time_string = '%d' % int(time.time())
        lc_name = 'lc-%s' % time_string
        lc = LaunchConfiguration(name=lc_name, image_id='ami-2272864b', instance_type='t1.micro', ebs_optimized=True)
        c.create_launch_configuration(lc)
        self.addCleanup(c.delete_launch_configuration, lc_name)