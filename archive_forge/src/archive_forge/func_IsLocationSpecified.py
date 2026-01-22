from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.core import properties
def IsLocationSpecified(args, resource_name):
    """Returns true if location is specified."""
    location_in_resource_name = '/locations/' in resource_name
    if args.IsKnownAndSpecified('location') and location_in_resource_name:
        raise errors.InvalidSCCInputError('Only provide location in a full resource name or in a --location flag, not both.')
    return args.IsKnownAndSpecified('location') or location_in_resource_name