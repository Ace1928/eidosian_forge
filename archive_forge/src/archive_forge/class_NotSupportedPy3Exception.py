from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class NotSupportedPy3Exception(exceptions.Error):
    """Commands that do not support python3."""