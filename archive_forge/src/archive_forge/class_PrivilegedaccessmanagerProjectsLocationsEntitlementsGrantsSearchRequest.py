from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivilegedaccessmanagerProjectsLocationsEntitlementsGrantsSearchRequest(_messages.Message):
    """A
  PrivilegedaccessmanagerProjectsLocationsEntitlementsGrantsSearchRequest
  object.

  Enums:
    CallerRelationshipValueValuesEnum: Required. Only Grants which the caller
      is related to by this relationship will be returned in the response.

  Fields:
    callerRelationship: Required. Only Grants which the caller is related to
      by this relationship will be returned in the response.
    filter: Optional. Only Grants matching this filter will be returned in the
      response.
    pageSize: Optional. Requested page size. Server may return fewer items
      than requested. If unspecified, server will pick an appropriate default.
    pageToken: Optional. A token identifying a page of results the server
      should return.
    parent: Required. The parent which owns the Grant resources.
  """

    class CallerRelationshipValueValuesEnum(_messages.Enum):
        """Required. Only Grants which the caller is related to by this
    relationship will be returned in the response.

    Values:
      CALLER_RELATIONSHIP_TYPE_UNSPECIFIED: Unspecified caller relationship
        type.
      HAD_CREATED: The user had created this Grant by calling CreateGrant
        earlier.
      CAN_APPROVE: The user is an Approver for the Entitlement that this Grant
        is parented under and can currently approve/deny it.
      HAD_APPROVED: The caller had successfully approved/denied this Grant
        earlier.
    """
        CALLER_RELATIONSHIP_TYPE_UNSPECIFIED = 0
        HAD_CREATED = 1
        CAN_APPROVE = 2
        HAD_APPROVED = 3
    callerRelationship = _messages.EnumField('CallerRelationshipValueValuesEnum', 1)
    filter = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)