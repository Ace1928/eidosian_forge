from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatastoreProjectsReserveIdsRequest(_messages.Message):
    """A DatastoreProjectsReserveIdsRequest object.

  Fields:
    projectId: Required. The ID of the project against which to make the
      request.
    reserveIdsRequest: A ReserveIdsRequest resource to be passed as the
      request body.
  """
    projectId = _messages.StringField(1, required=True)
    reserveIdsRequest = _messages.MessageField('ReserveIdsRequest', 2)