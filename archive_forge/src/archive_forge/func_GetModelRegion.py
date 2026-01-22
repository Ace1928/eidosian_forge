from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ml_engine import models
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.ml_engine import region_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def GetModelRegion(args):
    """Extract the region from the command line args.

  Args:
    args: arguments from parser.

  Returns:
    region, model_regions

    region: str, regional endpoint or global endpoint.
    model_regions: list, region where the model will be deployed.

  Raises:
    RegionArgError: if both --region and --regions are specified.
  """
    if args.IsSpecified('region') and args.IsSpecified('regions'):
        raise RegionArgError('Only one of --region or --regions can be specified.')
    if args.IsSpecified('regions'):
        return ('global', args.regions)
    if args.IsSpecified('region') and args.region != 'global':
        return (args.region, [args.region])
    region = region_util.GetRegion(args)
    if region != 'global':
        return (region, [region])
    log.warning('To specify a region where the model will deployed on the global endpoint, please use `--regions` and do not specify `--region`. Using [us-central1] by default on https://ml.googleapis.com. Please note that your model will be inaccessible from https://us-central1-ml.googelapis.com\n\nLearn more about regional endpoints and see a list of available regions: https://cloud.google.com/ai-platform/prediction/docs/regional-endpoints')
    return ('global', ['us-central1'])