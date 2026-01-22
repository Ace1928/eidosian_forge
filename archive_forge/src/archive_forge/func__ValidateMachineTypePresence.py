from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.instances import flags
def _ValidateMachineTypePresence(self, args):
    if not args.IsSpecified('custom_cpu') and (not args.IsSpecified('custom_memory')) and (not args.IsSpecified('machine_type')):
        raise calliope_exceptions.OneOfArgumentsRequiredException(['--custom-cpu', '--custom-memory', '--machine-type'], 'One of --custom-cpu, --custom-memory, --machine-type must be specified.')