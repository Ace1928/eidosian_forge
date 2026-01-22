from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
def ValidateCloudFunctionsURIs(unused_ref, args, request):
    """Checks if all provided Cloud Functions URIs are in correct format."""
    flags = ['source_cloud_function']
    function_pattern = re.compile('projects/(?:[a-z][a-z0-9-\\.:]*[a-z0-9])/locations/[-\\w]+/functions/[-\\w]+')
    for flag in flags:
        if not args.IsSpecified(flag):
            continue
        function = getattr(args, flag)
        if not function_pattern.match(function):
            raise InvalidInputError('Invalid value for flag {}: {}\nExpected Cloud Function in the following format:\n  projects/my-project/locations/location/functions/my-function'.format(flag, function))
    return request