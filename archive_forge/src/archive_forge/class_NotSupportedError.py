from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class NotSupportedError(exceptions.Error):
    """Error indicating that feature is not supported."""
    pass