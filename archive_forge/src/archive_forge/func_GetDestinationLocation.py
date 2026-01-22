from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.eventarc import triggers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.eventarc import flags
from googlecloudsdk.command_lib.eventarc import types
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
def GetDestinationLocation(self, args, trigger_ref, location_arg_name, destination_type):
    if not args.IsSpecified(location_arg_name):
        destination_location = trigger_ref.Parent().Name()
        if destination_location == 'global':
            raise NoDestinationLocationSpecifiedError('The `{}` flag is required when creating a global trigger with a destination {}.'.format(args.GetFlag(location_arg_name), destination_type))
    else:
        destination_location = args.GetValue(location_arg_name)
    return destination_location