from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthorizedNetwork(_messages.Message):
    """AuthorizedNetwork contains metadata for an authorized network.

  Fields:
    cidrRange: CIDR range for one authorzied network of the instance.
  """
    cidrRange = _messages.StringField(1)