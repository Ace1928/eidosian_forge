from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
class NetworkManagementError(exceptions.Error):
    """Top-level exception for all Network Management errors."""