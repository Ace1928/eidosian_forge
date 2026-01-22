from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryTestablePermissionsRequest(_messages.Message):
    """A request to get permissions which can be tested on a resource.

  Fields:
    fullResourceName: Required. The full resource name to query from the list
      of testable permissions. The name follows the Google Cloud Platform
      resource format. For example, a Cloud Platform project with id `my-
      project` will be named
      `//cloudresourcemanager.googleapis.com/projects/my-project`.
    pageSize: Optional limit on the number of permissions to include in the
      response. The default is 100, and the maximum is 1,000.
    pageToken: Optional pagination token returned in an earlier
      QueryTestablePermissionsRequest.
  """
    fullResourceName = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)