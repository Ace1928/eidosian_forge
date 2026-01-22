from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2InfoTypeTransformation(_messages.Message):
    """A transformation to apply to text that is identified as a specific
  info_type.

  Fields:
    infoTypes: InfoTypes to apply the transformation to. An empty list will
      cause this transformation to apply to all findings that correspond to
      infoTypes that were requested in `InspectConfig`.
    primitiveTransformation: Required. Primitive transformation to apply to
      the infoType.
  """
    infoTypes = _messages.MessageField('GooglePrivacyDlpV2InfoType', 1, repeated=True)
    primitiveTransformation = _messages.MessageField('GooglePrivacyDlpV2PrimitiveTransformation', 2)