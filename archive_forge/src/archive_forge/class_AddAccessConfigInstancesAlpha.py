from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags
@base.UniverseCompatible
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class AddAccessConfigInstancesAlpha(AddAccessConfigInstances):
    """Create a Compute Engine virtual machine access configuration."""
    _support_public_dns = True