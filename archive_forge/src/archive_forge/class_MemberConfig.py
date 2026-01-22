from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MemberConfig(_messages.Message):
    """The configuration for a member/cluster

  Fields:
    authMethods: A member may support multiple auth methods.
  """
    authMethods = _messages.MessageField('AuthMethod', 1, repeated=True)