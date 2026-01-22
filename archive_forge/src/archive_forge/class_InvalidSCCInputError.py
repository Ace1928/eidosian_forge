from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions as core_exceptions
class InvalidSCCInputError(core_exceptions.Error):
    """Exception raised for errors in the input."""