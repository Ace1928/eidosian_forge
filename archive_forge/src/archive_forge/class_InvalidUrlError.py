from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import six
class InvalidUrlError(Exception):
    """Exception raised when URL is invalid."""

    def __init__(self, message):
        Exception.__init__(self, message)
        self.message = message

    def __repr__(self):
        return str(self)

    def __str__(self):
        return 'InvalidUrlError: %s' % self.message