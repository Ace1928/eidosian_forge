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
def ValidateCustomHostnames(args):
    """Validates args supplied to --per-instance-hostnames."""
    if args.IsKnownAndSpecified('per_instance_hostnames'):
        if not args.IsKnownAndSpecified('predefined_names'):
            raise exceptions.RequiredArgumentException('--per-instance-hostnames', 'The `--per-instance-hostnames` argument must be used alongside the `--predefined-names` argument.')
        for instance_name, _ in args.per_instance_hostnames.items():
            if instance_name not in args.predefined_names:
                raise exceptions.InvalidArgumentException('--per-instance-hostnames', 'Instance [{}] missing in predefined_names. Instance names from --per-instance-hostnames must be included in --predefined-names flag.'.format(instance_name))