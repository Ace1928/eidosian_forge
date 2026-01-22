from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import exceptions
class NoMatchError(DebugError):
    """No object matched the search criteria."""

    def __init__(self, object_type, pattern=None):
        if pattern:
            super(NoMatchError, self).__init__('No {0} matched the pattern "{1}"'.format(object_type, pattern))
        else:
            super(NoMatchError, self).__init__('No {0} was found for this project.'.format(object_type))