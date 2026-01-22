from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsSessionsCreateRequest(_messages.Message):
    """A DataprocProjectsLocationsSessionsCreateRequest object.

  Fields:
    parent: Required. The parent resource where this session will be created.
    requestId: Optional. A unique ID used to identify the request. If the
      service receives two CreateSessionRequests (https://cloud.google.com/dat
      aproc/docs/reference/rpc/google.cloud.dataproc.v1#google.cloud.dataproc.
      v1.CreateSessionRequest)s with the same ID, the second request is
      ignored, and the first Session is created and stored in the
      backend.Recommendation: Set this value to a UUID
      (https://en.wikipedia.org/wiki/Universally_unique_identifier).The value
      must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). The maximum length is 40 characters.
    session: A Session resource to be passed as the request body.
    sessionId: Required. The ID to use for the session, which becomes the
      final component of the session's resource name.This value must be 4-63
      characters. Valid characters are /a-z-/.
  """
    parent = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)
    session = _messages.MessageField('Session', 3)
    sessionId = _messages.StringField(4)