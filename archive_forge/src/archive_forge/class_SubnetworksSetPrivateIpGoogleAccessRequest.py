from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SubnetworksSetPrivateIpGoogleAccessRequest(_messages.Message):
    """A SubnetworksSetPrivateIpGoogleAccessRequest object.

  Fields:
    privateIpGoogleAccess: A boolean attribute.
  """
    privateIpGoogleAccess = _messages.BooleanField(1)