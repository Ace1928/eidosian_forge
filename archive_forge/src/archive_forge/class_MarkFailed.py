from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.recommender import recommendation
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.recommender import flags
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class MarkFailed(base.Command):
    """Mark a recommendation's state as FAILED.

      Mark a recommendation's state as FAILED. Can be applied to recommendations
      in ACTIVE, CLAIMED, SUCCEEDED, or FAILED state.
  """
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Args:
      parser: An argparse parser that you can use to add arguments that go on
        the command line after this command.
    """
        flags.AddParentFlagsToParser(parser)
        parser.add_argument('RECOMMENDATION', type=str, help='Recommendation id which will be marked as FAILED.')
        parser.add_argument('--location', metavar='LOCATION', required=True, help='Location.')
        parser.add_argument('--recommender', metavar='RECOMMENDER', required=True, help='Recommender of recommendation.')
        parser.add_argument('--etag', metavar='ETAG', required=True, help='Etag of a recommendation.')
        parser.add_argument('--state-metadata', type=arg_parsers.ArgDict(min_length=1), default={}, help='State metadata for recommendation, in format of --state-metadata=key1=value1,key2=value2', metavar='KEY=VALUE', action=arg_parsers.StoreOnceAction)

    def Run(self, args):
        """Run 'gcloud recommender recommendations mark-failed'.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Returns:
      The recommendations after being marked as failed.
    """
        client = recommendation.CreateClient(self.ReleaseTrack())
        name = flags.GetRecommendationName(args)
        return client.MarkFailed(name, args.state_metadata, args.etag)