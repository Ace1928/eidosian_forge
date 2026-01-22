from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
import six
class ResponseNotJSONError(RequestError):
    """Raised when a request to the Apigee API returns a malformed response."""

    def __init__(self, error, resource_type=None, resource_identifier=None, body=None, user_help=None):
        if all((hasattr(error, attr) for attr in ['msg', 'lineno', 'colno'])):
            reason = '%s at %d:%d' % (error.msg, error.lineno, error.colno)
        else:
            reason = six.text_type(error)
        super(ResponseNotJSONError, self).__init__(resource_type, resource_identifier, 'parse', reason, json.dumps({'response': body}), user_help=user_help)
        self.base_error = error

    def RewrittenError(self, resource_type, method):
        """Returns a copy of the error with a new resource type."""
        body = self.details['response'] if self.details else None
        return type(self)(self.base_error, resource_type, self.resource_identifier, body=body, user_help=self.user_help)