from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class CloudShell(base.Group):
    """Manage Google Cloud Shell."""
    category = base.MANAGEMENT_TOOLS_CATEGORY
    detailed_help = {'DESCRIPTION': '          Interact with and connect to your Cloud Shell environment.\n\n          More information on Cloud Shell can be found at\n          https://cloud.google.com/shell/docs/.\n          ', 'NOTES': textwrap.dedent('          The previous *gcloud alpha shell* command to launch an interactive\n          shell was renamed to *gcloud alpha interactive*.\n          ')}

    @staticmethod
    def Args(parser):
        pass

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args
        base.DisableUserProjectQuota()