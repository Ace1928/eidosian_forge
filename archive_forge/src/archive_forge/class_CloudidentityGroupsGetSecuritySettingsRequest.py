from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityGroupsGetSecuritySettingsRequest(_messages.Message):
    """A CloudidentityGroupsGetSecuritySettingsRequest object.

  Fields:
    name: Required. The security settings to retrieve. Format:
      `groups/{group_id}/securitySettings`
    readMask: Field-level read mask of which fields to return. "*" returns all
      fields. If not specified, all fields will be returned. May only contain
      the following field: `member_restriction`.
  """
    name = _messages.StringField(1, required=True)
    readMask = _messages.StringField(2)