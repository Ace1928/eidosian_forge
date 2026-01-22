from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OperationDnsKeyContext(_messages.Message):
    """A OperationDnsKeyContext object.

  Fields:
    newValue: The post-operation DnsKey resource.
    oldValue: The pre-operation DnsKey resource.
  """
    newValue = _messages.MessageField('DnsKey', 1)
    oldValue = _messages.MessageField('DnsKey', 2)