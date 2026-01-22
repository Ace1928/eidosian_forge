from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourcesListResponse(_messages.Message):
    """A response containing a partial list of resources and a page token used
  to build the next request if the request has been truncated.

  Fields:
    nextPageToken: A token used to continue a truncated list request.
    resources: Resources contained in this list response.
  """
    nextPageToken = _messages.StringField(1)
    resources = _messages.MessageField('Resource', 2, repeated=True)