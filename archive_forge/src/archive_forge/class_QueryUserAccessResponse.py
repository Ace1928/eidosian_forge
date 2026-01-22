from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryUserAccessResponse(_messages.Message):
    """Request message for QueryUserAccess method.

  Fields:
    accessibleVisibilityLabels: Any visibility labels on the service that are
      accessible by the user.
    canAccessService: True if the user can access the service and any
      unrestricted API surface.
  """
    accessibleVisibilityLabels = _messages.StringField(1, repeated=True)
    canAccessService = _messages.BooleanField(2)