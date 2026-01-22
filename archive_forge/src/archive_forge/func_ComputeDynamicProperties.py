from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import instance_groups_utils
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers
def ComputeDynamicProperties(self, args, items, holder):
    return instance_groups_utils.ComputeInstanceGroupManagerMembership(compute_holder=holder, items=items, filter_mode=instance_groups_utils.InstanceGroupFilteringMode.ALL_GROUPS)