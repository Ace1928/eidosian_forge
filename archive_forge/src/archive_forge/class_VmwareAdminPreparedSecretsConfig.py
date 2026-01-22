from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareAdminPreparedSecretsConfig(_messages.Message):
    """VmwareAdminPreparedSecretsConfig represents configuration for admin
  cluster prepared secrets.

  Fields:
    enabled: Whether prepared secrets is enabled.
  """
    enabled = _messages.BooleanField(1)