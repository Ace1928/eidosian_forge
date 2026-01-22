from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as instance_groups_managed_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
import six
def ValidateIgmReference(igm_ref):
    if igm_ref.Collection() not in ['compute.instanceGroupManagers', 'compute.regionInstanceGroupManagers']:
        raise ValueError('Unknown reference type {0}'.format(igm_ref.Collection()))