from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RetrieveTransferParametersResponse(_messages.Message):
    """Deprecated: For more information, see [Cloud Domains feature
  deprecation](https://cloud.google.com/domains/docs/deprecations/feature-
  deprecations). Response for the `RetrieveTransferParameters` method.

  Fields:
    transferParameters: Parameters to use when calling the `TransferDomain`
      method.
  """
    transferParameters = _messages.MessageField('TransferParameters', 1)