from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import instance_groups_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.core import properties
class GetNamedPorts(base.ListCommand):
    """Implements get-named-ports command, alpha, and beta versions."""

    @staticmethod
    def Args(parser):
        instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_ARG.AddArgument(parser)
        parser.display_info.AddFormat('table(name, port)')

    def Run(self, args):
        """Retrieves response with named ports."""
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        project = properties.VALUES.core.project.Get(required=True)
        group_ref = instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_ARG.ResolveAsResource(args, holder.resources, default_scope=compute_scope.ScopeEnum.ZONE, scope_lister=flags.GetDefaultScopeLister(holder.client, project))
        return instance_groups_utils.OutputNamedPortsForGroup(group_ref, holder.client)
    detailed_help = instance_groups_utils.INSTANCE_GROUP_GET_NAMED_PORT_DETAILED_HELP