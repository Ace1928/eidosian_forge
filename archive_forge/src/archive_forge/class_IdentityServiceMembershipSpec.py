from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdentityServiceMembershipSpec(_messages.Message):
    """**Anthos Identity Service**: Configuration for a single Membership.

  Fields:
    authMethods: A member may support multiple auth methods.
  """
    authMethods = _messages.MessageField('IdentityServiceAuthMethod', 1, repeated=True)