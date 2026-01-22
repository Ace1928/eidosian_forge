from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.service_directory import services
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.service_directory import flags
from googlecloudsdk.command_lib.service_directory import resource_args
@base.ReleaseTracks(base.ReleaseTrack.GA)
class Resolve(base.Command):
    """Resolves a service."""
    detailed_help = {'EXAMPLES': '          To resolve Service Directory services, run:\n\n            $ {command} my-service --namespace=my-namespace --location=us-east1\n          '}

    @staticmethod
    def Args(parser):
        resource_args.AddServiceResourceArg(parser, 'to resolve.')
        flags.AddMaxEndpointsFlag(parser)
        flags.AddEndpointFilterFlag(parser)

    def Run(self, args):
        client = services.ServicesClient(self.GetReleaseTrack())
        service_ref = args.CONCEPTS.service.Parse()
        return client.Resolve(service_ref, args.max_endpoints, args.endpoint_filter)

    def GetReleaseTrack(self):
        return base.ReleaseTrack.GA