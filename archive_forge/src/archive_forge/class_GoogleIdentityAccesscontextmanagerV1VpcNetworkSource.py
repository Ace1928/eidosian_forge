from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIdentityAccesscontextmanagerV1VpcNetworkSource(_messages.Message):
    """The originating network source in Google Cloud.

  Fields:
    vpcSubnetwork: Sub-segment ranges of a VPC network.
  """
    vpcSubnetwork = _messages.MessageField('GoogleIdentityAccesscontextmanagerV1VpcSubNetwork', 1)