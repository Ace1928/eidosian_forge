from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.compute.resource_policies import flags as maintenance_flags
from googlecloudsdk.command_lib.util.args import labels_util
def ValidateMaxCountPerZoneArgs(args):
    """Validates args supplied to --max-count-per-zone."""
    if args.IsKnownAndSpecified('max_count_per_zone'):
        for zone, count in args.max_count_per_zone.items():
            if not ValidateZone(zone):
                raise exceptions.InvalidArgumentException('--max-count-per-zone', 'Key [{}] must be a zone.'.format(zone))
            if not ValidateNaturalCount(count):
                raise exceptions.InvalidArgumentException('--max-count-per-zone', 'Value [{}] must be a positive natural number.'.format(count))