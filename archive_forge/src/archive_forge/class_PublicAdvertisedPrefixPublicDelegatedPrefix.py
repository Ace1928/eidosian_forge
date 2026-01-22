from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PublicAdvertisedPrefixPublicDelegatedPrefix(_messages.Message):
    """Represents a CIDR range which can be used to assign addresses.

  Fields:
    ipRange: The IP address range of the public delegated prefix
    name: The name of the public delegated prefix
    project: The project number of the public delegated prefix
    region: The region of the public delegated prefix if it is regional. If
      absent, the prefix is global.
    status: The status of the public delegated prefix. Possible values are:
      INITIALIZING: The public delegated prefix is being initialized and
      addresses cannot be created yet. ANNOUNCED: The public delegated prefix
      is active.
  """
    ipRange = _messages.StringField(1)
    name = _messages.StringField(2)
    project = _messages.StringField(3)
    region = _messages.StringField(4)
    status = _messages.StringField(5)