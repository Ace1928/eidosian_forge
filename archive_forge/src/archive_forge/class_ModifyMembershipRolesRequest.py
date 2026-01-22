from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ModifyMembershipRolesRequest(_messages.Message):
    """The request message for MembershipsService.ModifyMembershipRoles.

  Fields:
    addRoles: The `MembershipRole`s to be added. Adding or removing roles in
      the same request as updating roles is not supported. Must not be set if
      `update_roles_params` is set.
    removeRoles: The `name`s of the `MembershipRole`s to be removed. Adding or
      removing roles in the same request as updating roles is not supported.
      It is not possible to remove the `MEMBER` `MembershipRole`. If you wish
      to delete a `Membership`, call MembershipsService.DeleteMembership
      instead. Must not contain `MEMBER`. Must not be set if
      `update_roles_params` is set.
    updateRolesParams: The `MembershipRole`s to be updated. Updating roles in
      the same request as adding or removing roles is not supported. Must not
      be set if either `add_roles` or `remove_roles` is set.
  """
    addRoles = _messages.MessageField('MembershipRole', 1, repeated=True)
    removeRoles = _messages.StringField(2, repeated=True)
    updateRolesParams = _messages.MessageField('UpdateMembershipRolesParams', 3, repeated=True)