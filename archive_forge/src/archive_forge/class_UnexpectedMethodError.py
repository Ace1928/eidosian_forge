from __future__ import absolute_import
import json
import six
from googleapiclient import _helpers as util
class UnexpectedMethodError(Error):
    """Exception raised by RequestMockBuilder on unexpected calls."""

    @util.positional(1)
    def __init__(self, methodId=None):
        """Constructor for an UnexpectedMethodError."""
        super(UnexpectedMethodError, self).__init__('Received unexpected call %s' % methodId)