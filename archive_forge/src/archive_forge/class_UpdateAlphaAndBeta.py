from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import update_util
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class UpdateAlphaAndBeta(base.UpdateCommand):
    """Updates an App Engine application(Alpha and Beta version)."""
    detailed_help = _DETAILED_HELP

    @staticmethod
    def Args(parser):
        update_util.AddAppUpdateFlags(parser)

    def Run(self, args):
        update_util.PatchApplication(self.ReleaseTrack(), split_health_checks=args.split_health_checks, service_account=args.service_account)