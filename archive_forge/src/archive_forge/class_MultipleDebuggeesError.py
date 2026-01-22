from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import exceptions
class MultipleDebuggeesError(DebugError):
    """Multiple targets matched the search criteria."""

    def __init__(self, pattern, debuggees):
        if pattern:
            pattern_msg = ' matching "{0}"'.format(pattern)
        else:
            pattern_msg = ''
        super(MultipleDebuggeesError, self).__init__('Multiple possible targets found{0}.\nUse the --target option to select one of the following targets:\n    {1}\n'.format(pattern_msg, '\n    '.join([d.name for d in debuggees])))