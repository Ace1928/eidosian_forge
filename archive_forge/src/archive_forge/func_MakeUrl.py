from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.resource_policies import util as maintenance_util
from googlecloudsdk.core.util import times
import six
def MakeUrl(resources, value, reservation_ref):
    return maintenance_util.ParseResourcePolicyWithZone(resources, value, project=reservation_ref.project, zone=reservation_ref.zone).SelfLink()