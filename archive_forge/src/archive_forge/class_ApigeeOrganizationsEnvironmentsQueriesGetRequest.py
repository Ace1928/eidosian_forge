from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsQueriesGetRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsQueriesGetRequest object.

  Fields:
    name: Required. Name of the asynchronous query to get. Must be of the form
      `organizations/{org}/environments/{env}/queries/{queryId}`.
  """
    name = _messages.StringField(1, required=True)