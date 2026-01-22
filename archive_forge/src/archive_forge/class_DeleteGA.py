from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.apphub import utils as api_lib_utils
from googlecloudsdk.api_lib.apphub.applications import services as apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.apphub import flags
@base.ReleaseTracks(base.ReleaseTrack.GA)
class DeleteGA(base.DeleteCommand):
    """Delete an Apphub application service."""
    detailed_help = _DETAILED_HELP

    @staticmethod
    def Args(parser):
        flags.AddDeleteApplicationServiceFlags(parser)

    def Run(self, args):
        """Run the delete command."""
        client = apis.ServicesClient(release_track=base.ReleaseTrack.GA)
        service_ref = api_lib_utils.GetApplicationServiceRef(args)
        return client.Delete(service=service_ref.RelativeName(), async_flag=args.async_)