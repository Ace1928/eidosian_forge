from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.instance_groups.managed import wait_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
@base.ReleaseTracks(base.ReleaseTrack.GA)
class WaitUntilGA(base.Command):
    """Wait until the managed instance group reaches the desired state."""

    @staticmethod
    def Args(parser):
        _AddArgs(parser=parser)

    def CreateGroupReference(self, client, resources, args):
        return instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG.ResolveAsResource(args, resources, default_scope=compute_scope.ScopeEnum.ZONE, scope_lister=flags.GetDefaultScopeLister(client))

    def Run(self, args):
        """Issues requests necessary to wait until stable on a MIG."""
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        group_ref = self.CreateGroupReference(client, holder.resources, args)
        if args.stable:
            igm_state = wait_utils.IgmState.STABLE
        elif args.version_target_reached:
            igm_state = wait_utils.IgmState.VERSION_TARGET_REACHED
        elif args.all_instances_config_effective:
            igm_state = wait_utils.IgmState.ALL_INSTANCES_CONFIG_EFFECTIVE
        wait_utils.WaitForIgmState(client, group_ref, igm_state, args.timeout)