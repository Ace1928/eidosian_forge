from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateConsumerConfigRequest(_messages.Message):
    """Request to update the configuration of a service networking connection
  including the import/export of custom routes and subnetwork routes with
  public IP.

  Fields:
    consumerConfig: Required. The updated peering config.
  """
    consumerConfig = _messages.MessageField('ConsumerConfig', 1)