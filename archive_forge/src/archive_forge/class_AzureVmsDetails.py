from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AzureVmsDetails(_messages.Message):
    """AzureVmsDetails describes VMs in Azure.

  Fields:
    details: The details of the Azure VMs.
  """
    details = _messages.MessageField('AzureVmDetails', 1, repeated=True)