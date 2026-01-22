from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.auth import auth_util
from googlecloudsdk.command_lib.resource_manager import completers
class SetQuotaProject(base.SilentCommand):
    """Update or add a quota project in application default credentials (ADC).

  A quota project is a Google Cloud Project that will be used for billing
  and quota limits.

  Before running this command, an ADC must already be generated using
  $ gcloud auth application-default login.
  The quota project defined in the ADC will be used by the Google client
  libraries.
  The existing application default credentials must have the
  `serviceusage.services.use` permission on the given project.

  ## EXAMPLES

  To update the quota project in application default credentials to
  `my-quota-project`, run:

    $ {command} my-quota-project
  """

    @staticmethod
    def Args(parser):
        base.Argument('quota_project_id', metavar='QUOTA_PROJECT_ID', completer=completers.ProjectCompleter, help='Quota project ID to add to application default credentials. If a quota project already exists, it will be updated.').AddToParser(parser)

    def Run(self, args):
        return auth_util.AddQuotaProjectToADC(args.quota_project_id)