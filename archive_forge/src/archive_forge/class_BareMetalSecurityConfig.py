from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalSecurityConfig(_messages.Message):
    """Specifies the security related settings for the bare metal user cluster.

  Fields:
    authorization: Configures user access to the user cluster.
  """
    authorization = _messages.MessageField('Authorization', 1)