from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
def ValidateInstanceNames(unused_ref, args, request):
    """Checks if all provided instances are in valid format."""
    flags = ['source_instance', 'destination_instance']
    instance_pattern = re.compile('projects/(?:[a-z][a-z0-9-\\.:]*[a-z0-9])/zones/[-\\w]+/instances/[-\\w]+')
    for flag in flags:
        if args.IsSpecified(flag):
            instance = getattr(args, flag)
            if not instance_pattern.match(instance):
                raise InvalidInputError('Invalid value for flag {}: {}\nExpected instance in the following format:\n  projects/my-project/zones/zone/instances/my-instance'.format(flag, instance))
    return request