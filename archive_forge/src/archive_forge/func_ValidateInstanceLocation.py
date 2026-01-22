from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import exceptions as sql_exceptions
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.calliope import exceptions
def ValidateInstanceLocation(args):
    """Construct a Cloud SQL instance from command line args.

  Args:
    args: argparse.Namespace, The CLI arg namespace.

  Raises:
    RequiredArgumentException: Zone is required.
    ConflictingArgumentsException: Zones in arguments belong to different
    regions.
  """
    if args.IsSpecified('secondary_zone') and (not args.IsSpecified('zone')):
        raise exceptions.RequiredArgumentException('--zone', '`--zone` is required if --secondary-zone is used while creating an instance.')
    if args.IsSpecified('secondary_zone') and args.IsSpecified('zone'):
        if args.zone == args.secondary_zone:
            raise exceptions.ConflictingArgumentsException('Zones in arguments --zone and --secondary-zone are identical.')
        region_from_zone = api_util.GetRegionFromZone(args.zone)
        region_from_secondary_zone = api_util.GetRegionFromZone(args.secondary_zone)
        if region_from_zone != region_from_secondary_zone:
            raise exceptions.ConflictingArgumentsException('Zones in arguments --zone and --secondary-zone belong to different regions.')