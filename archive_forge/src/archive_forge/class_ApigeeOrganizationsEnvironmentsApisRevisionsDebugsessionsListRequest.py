from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsApisRevisionsDebugsessionsListRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsApisRevisionsDebugsessionsListRequest
  object.

  Fields:
    pageSize: Maximum number of debug sessions to return. The page size
      defaults to 25.
    pageToken: Page token, returned from a previous ListDebugSessions call,
      that you can use to retrieve the next page.
    parent: Required. The name of the API Proxy revision deployment for which
      to list debug sessions. Must be of the form: `organizations/{organizatio
      n}/environments/{environment}/apis/{api}/revisions/{revision}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)