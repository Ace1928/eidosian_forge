from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterUser(_messages.Message):
    """ClusterUser configures user principals for an RBAC policy.

  Fields:
    username: Required. The name of the user, e.g. `my-gcp-id@gmail.com`.
  """
    username = _messages.StringField(1)