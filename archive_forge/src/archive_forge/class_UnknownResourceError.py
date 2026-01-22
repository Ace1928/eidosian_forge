from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.exceptions import Error
import six
class UnknownResourceError(Error):
    """Raised when a instant snapshot resource argument is neither regional nor zonal."""