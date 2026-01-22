from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import functools
import ipaddress
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import disks_util
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.zones import service as zones_service
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as core_resources
import six
def ValidateInstanceScheduling(args, support_max_run_duration=False):
    """Validates instance scheduling related flags."""
    if args.IsSpecified('instance_termination_action'):
        if support_max_run_duration:
            if not (args.IsSpecified('provisioning_model') or args.IsSpecified('max_run_duration') or args.IsSpecified('termination_time')):
                raise exceptions.MinimumArgumentException(['--provisioning-model', '--max-run-duration', '--termination-time'], 'required with argument `--instance-termination-action`.')
        elif not args.IsSpecified('provisioning_model'):
            raise exceptions.RequiredArgumentException('--provisioning-model', 'required with argument `--instance-termination-action`.')
    if support_max_run_duration and args.IsSpecified('termination_time') and args.IsSpecified('max_run_duration'):
        raise compute_exceptions.ArgumentError('Must specify exactly one of --max-run-duration or --termination-time as these fields are mutually exclusive.')