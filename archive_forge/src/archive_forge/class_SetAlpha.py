from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import forwarding_rules_utils as utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.forwarding_rules import flags
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class SetAlpha(SetBeta):
    """Modify a forwarding rule to direct network traffic to a new target."""
    _include_regional_tcp_proxy = True