from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.workstations import workstations
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.workstations import flags as workstations_flags
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class StartTcpTunnel(base.Command):
    """Start a tunnel through which a local process can forward TCP traffic to the workstation."""
    detailed_help = {'DESCRIPTION': '{description}', 'EXAMPLES': '          To start a tunnel to port 22 on a workstation, run:\n\n            $ {command} --project=my-project --region=us-central1 --cluster=my-cluster --config=my-config my-workstation 22\n          '}

    @staticmethod
    def Args(parser):
        workstations_flags.AddWorkstationResourceArg(parser)
        workstations_flags.AddWorkstationPortField(parser)
        workstations_flags.AddLocalHostPortField(parser)

    def Run(self, args):
        client = workstations.Workstations(self.ReleaseTrack())
        client.StartTcpTunnel(args)