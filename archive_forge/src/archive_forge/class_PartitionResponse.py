from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PartitionResponse(_messages.Message):
    """The response for PartitionQuery or PartitionRead

  Fields:
    partitions: Partitions created by this request.
    transaction: Transaction created by this request.
  """
    partitions = _messages.MessageField('Partition', 1, repeated=True)
    transaction = _messages.MessageField('Transaction', 2)