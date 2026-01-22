from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImportProductSetsRequest(_messages.Message):
    """Request message for the `ImportProductSets` method.

  Fields:
    inputConfig: Required. The input content for the list of requests.
  """
    inputConfig = _messages.MessageField('ImportProductSetsInputConfig', 1)