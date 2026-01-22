from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
import six
class MissingIdentifierError(exceptions.Error):
    """Raised when a request to the Apigee API is missing an expected identifier.

  Normally this situation should be caught by a required argument being missing
  or similar; this error is a fallback in case a corner case slips past those
  higher level checks.
  """

    def __init__(self, name):
        message = 'Command requires a %s but no %s was provided.' % (name, name)
        super(MissingIdentifierError, self).__init__(message)