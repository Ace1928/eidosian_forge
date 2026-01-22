from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class CannotOpenFileError(exceptions.Error):
    """Cannot open file."""

    def __init__(self, f, e):
        super(CannotOpenFileError, self).__init__('Failed to open file [{f}]: {e}'.format(f=f, e=e))