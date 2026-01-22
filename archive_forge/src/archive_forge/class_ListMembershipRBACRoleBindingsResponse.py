from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListMembershipRBACRoleBindingsResponse(_messages.Message):
    """List of Membership RBACRoleBindings.

  Fields:
    nextPageToken: A token to request the next page of resources from the
      `ListMembershipRBACRoleBindings` method. The value of an empty string
      means that there are no more resources to return.
    rbacrolebindings: The list of Membership RBACRoleBindings.
  """
    nextPageToken = _messages.StringField(1)
    rbacrolebindings = _messages.MessageField('RBACRoleBinding', 2, repeated=True)