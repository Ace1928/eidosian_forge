from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.recommender import base
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.args import common_args
def GetConfigsParentFromFlags(args, is_insight_api):
    """Parsing args for url string for recommender and insigh type configs apis.

  Args:
      args: argparse.Namespace, The arguments that this command was invoked
        with.
      is_insight_api: whether this is an insight api.

  Returns:
      The full url string based on flags given by user.
  """
    url = 'projects/{0}'.format(args.project)
    url = url + '/locations/{0}'.format(args.location)
    if is_insight_api:
        url = url + '/insightTypes/{0}'.format(args.insight_type)
    else:
        url = url + '/recommenders/{0}'.format(args.recommender)
    return url + '/config'