from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NextHopVpcNetwork(_messages.Message):
    """A NextHopVpcNetwork object.

  Fields:
    uri: The URI of the VPC network resource
  """
    uri = _messages.StringField(1)