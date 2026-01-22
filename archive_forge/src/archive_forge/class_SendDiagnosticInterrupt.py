from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags
@base.UniverseCompatible
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class SendDiagnosticInterrupt(base.SilentCommand):
    """Send a diagnostic interrupt to a virtual machine instance."""

    @staticmethod
    def Args(parser):
        flags.INSTANCE_ARG.AddArgument(parser)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        instance_ref = flags.INSTANCE_ARG.ResolveAsResource(args, holder.resources, scope_lister=flags.GetInstanceZoneScopeLister(client))
        request = client.messages.ComputeInstancesSendDiagnosticInterruptRequest(**instance_ref.AsDict())
        return client.MakeRequests([(client.apitools_client.instances, 'SendDiagnosticInterrupt', request)])