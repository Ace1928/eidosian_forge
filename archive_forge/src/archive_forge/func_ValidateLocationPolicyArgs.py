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
def ValidateLocationPolicyArgs(args):
    """Validates args supplied to --location-policy."""
    if args.IsSpecified('location_policy'):
        for zone, policy in args.location_policy.items():
            zone_split = zone.split('-')
            if len(zone_split) != 3 or (len(zone_split[2]) != 1 or not zone_split[2].isalpha()) or (not zone_split[1][-1].isdigit()):
                raise exceptions.InvalidArgumentException('--location-policy', 'Key [{}] must be a zone.'.format(zone))
            if policy not in ['allow', 'deny']:
                raise exceptions.InvalidArgumentException('--location-policy', 'Value [{}] must be one of [allow, deny]'.format(policy))