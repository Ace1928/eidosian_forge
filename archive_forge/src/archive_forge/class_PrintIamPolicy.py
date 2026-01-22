from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import flags
from googlecloudsdk.command_lib.artifacts import upgrade_util
from googlecloudsdk.command_lib.artifacts import util
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class PrintIamPolicy(base.Command):
    """Print an Artifact Registry IAM policy for Container Registry to Artifact Registry upgrade.

  Print an Artifact Registry IAM policy that is equivalent to the IAM policy
  applied to the storage bucket for the specified Container Registry hostname.
  Apply the returned policy to the Artifact Registry repository that will
  replace the specified host. If the project has an organization, this command
  analyzes IAM policies at the organization level. Otherwise, this command
  analyzes IAM policies at the project level. See required permissions at
  https://cloud.google.com/policy-intelligence/docs/analyze-iam-policies#required-permissions.
  """
    detailed_help = {'DESCRIPTION': '{description}', 'EXAMPLES': "  To print an equivalent Artifact Registry IAM policy for 'gcr.io/my-project':\n\n      $ {command} upgrade print-iam-policy gcr.io --project=my-project\n  "}

    @staticmethod
    def Args(parser):
        flags.GetGCRDomainArg().AddToParser(parser)

    def Run(self, args):
        """Runs the command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      An iam.Policy.
    """
        domain = args.DOMAIN
        project = util.GetProject(args)
        return upgrade_util.iam_policy(domain, project)