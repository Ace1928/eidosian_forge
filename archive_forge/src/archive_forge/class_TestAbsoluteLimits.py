import copy
from unittest import mock
from keystoneauth1 import adapter
from openstack.compute.v2 import limits
from openstack.tests.unit import base
class TestAbsoluteLimits(base.TestCase):

    def test_basic(self):
        sot = limits.AbsoluteLimits()
        self.assertIsNone(sot.resource_key)
        self.assertIsNone(sot.resources_key)
        self.assertEqual('', sot.base_path)
        self.assertFalse(sot.allow_create)
        self.assertFalse(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertFalse(sot.allow_delete)
        self.assertFalse(sot.allow_list)

    def test_make_it(self):
        sot = limits.AbsoluteLimits(**ABSOLUTE_LIMITS)
        self.assertEqual(ABSOLUTE_LIMITS['maxImageMeta'], sot.image_meta)
        self.assertEqual(ABSOLUTE_LIMITS['maxSecurityGroupRules'], sot.security_group_rules)
        self.assertEqual(ABSOLUTE_LIMITS['maxSecurityGroups'], sot.security_groups)
        self.assertEqual(ABSOLUTE_LIMITS['maxServerMeta'], sot.server_meta)
        self.assertEqual(ABSOLUTE_LIMITS['maxTotalCores'], sot.total_cores)
        self.assertEqual(ABSOLUTE_LIMITS['maxTotalFloatingIps'], sot.floating_ips)
        self.assertEqual(ABSOLUTE_LIMITS['maxTotalInstances'], sot.instances)
        self.assertEqual(ABSOLUTE_LIMITS['maxTotalKeypairs'], sot.keypairs)
        self.assertEqual(ABSOLUTE_LIMITS['maxTotalRAMSize'], sot.total_ram)
        self.assertEqual(ABSOLUTE_LIMITS['maxServerGroups'], sot.server_groups)
        self.assertEqual(ABSOLUTE_LIMITS['maxServerGroupMembers'], sot.server_group_members)
        self.assertEqual(ABSOLUTE_LIMITS['totalFloatingIpsUsed'], sot.floating_ips_used)
        self.assertEqual(ABSOLUTE_LIMITS['totalSecurityGroupsUsed'], sot.security_groups_used)
        self.assertEqual(ABSOLUTE_LIMITS['totalRAMUsed'], sot.total_ram_used)
        self.assertEqual(ABSOLUTE_LIMITS['totalInstancesUsed'], sot.instances_used)
        self.assertEqual(ABSOLUTE_LIMITS['totalServerGroupsUsed'], sot.server_groups_used)
        self.assertEqual(ABSOLUTE_LIMITS['totalCoresUsed'], sot.total_cores_used)