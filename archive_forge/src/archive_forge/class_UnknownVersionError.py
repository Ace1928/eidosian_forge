from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.core import exceptions
class UnknownVersionError(exceptions.Error):
    """Unable to find API version in APIs map."""

    def __init__(self, api_name, api_version):
        super(UnknownVersionError, self).__init__('The [{0}] API does not have version [{1}] in the APIs map'.format(api_name, api_version))