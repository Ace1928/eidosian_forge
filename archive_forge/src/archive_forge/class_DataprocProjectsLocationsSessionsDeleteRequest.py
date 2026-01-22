from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsSessionsDeleteRequest(_messages.Message):
    """A DataprocProjectsLocationsSessionsDeleteRequest object.

  Fields:
    name: Required. The name of the session resource to delete.
    requestId: Optional. A unique ID used to identify the request. If the
      service receives two DeleteSessionRequest (https://cloud.google.com/data
      proc/docs/reference/rpc/google.cloud.dataproc.v1#google.cloud.dataproc.v
      1.DeleteSessionRequest)s with the same ID, the second request is
      ignored.Recommendation: Set this value to a UUID
      (https://en.wikipedia.org/wiki/Universally_unique_identifier).The value
      must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). The maximum length is 40 characters.
  """
    name = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)