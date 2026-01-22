from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as instance_groups_managed_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import rolling_action
from googlecloudsdk.command_lib.compute.managed_instance_groups import update_instances_utils
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class StartUpdateBeta(StartUpdate):
    """Replaces instances in a managed instance group.

  Deletes the existing instance and creates a new instance from the target
  template. The Updater creates a brand new instance with all new instance
  properties, such as new internal and external IP addresses.
  """

    @staticmethod
    def Args(parser):
        _AddArgs(parser, supports_min_ready=True)
        instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG.AddArgument(parser)