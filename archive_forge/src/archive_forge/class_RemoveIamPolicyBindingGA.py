from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.apphub import utils as api_lib_utils
from googlecloudsdk.api_lib.apphub.applications import client as apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.apphub import flags
from googlecloudsdk.command_lib.iam import iam_util
@base.ReleaseTracks(base.ReleaseTrack.GA)
class RemoveIamPolicyBindingGA(base.Command):
    """Remove IAM policy binding from an Apphub application."""
    detailed_help = _DETAILED_HELP

    @staticmethod
    def Args(parser):
        flags.AddRemoveIamPolicyBindingFlags(parser)
        iam_util.AddArgsForRemoveIamPolicyBinding(parser)

    def Run(self, args):
        client = apis.ApplicationsClient(release_track=base.ReleaseTrack.GA)
        app_ref = api_lib_utils.GetApplicationRef(args)
        return client.RemoveIamPolicyBinding(app_id=app_ref.RelativeName(), member=args.member, role=args.role)