from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.recommender import recommendation
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.recommender import flags
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class MarkActive(base.Command):
    """Mark Active operations for a recommendation.

     Mark a recommendation's state as ACTIVE. Can be applied to recommendations
     in DISMISSED state. This currently supports the following parent entities:
     project, billing account, folder, and organization.

     ## EXAMPLES
     To mark a recommenation as ACTIVE:

      $ {command} RECOMMENDATION_ID --project=${PROJECT} --location=${LOCATION}
      --recommender=${RECOMMENDER} --etag=etag
  """

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Args:
      parser: An argparse parser that you can use to add arguments that go on
        the command line after this command.
    """
        flags.AddParentFlagsToParser(parser)
        parser.add_argument('RECOMMENDATION', type=str, help='Recommendation id which will be marked as active')
        parser.add_argument('--location', metavar='LOCATION', required=True, help='Location')
        parser.add_argument('--recommender', metavar='RECOMMENDER', required=True, help='Recommender of the recommendations')
        parser.add_argument('--etag', required=True, metavar='ETAG', help='Etag of a recommendation')

    def Run(self, args):
        """Run 'gcloud recommender recommendations mark-active'.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Returns:
      The result recommendations after being marked as active
    """
        client = recommendation.CreateClient(self.ReleaseTrack())
        name = flags.GetRecommendationName(args)
        return client.MarkActive(name, args.etag)