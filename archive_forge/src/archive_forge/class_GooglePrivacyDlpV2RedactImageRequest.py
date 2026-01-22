from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2RedactImageRequest(_messages.Message):
    """Request to search for potentially sensitive info in an image and redact
  it by covering it with a colored rectangle.

  Fields:
    byteItem: The content must be PNG, JPEG, SVG or BMP.
    imageRedactionConfigs: The configuration for specifying what content to
      redact from images.
    includeFindings: Whether the response should include findings along with
      the redacted image.
    inspectConfig: Configuration for the inspector.
    locationId: Deprecated. This field has no effect.
  """
    byteItem = _messages.MessageField('GooglePrivacyDlpV2ByteContentItem', 1)
    imageRedactionConfigs = _messages.MessageField('GooglePrivacyDlpV2ImageRedactionConfig', 2, repeated=True)
    includeFindings = _messages.BooleanField(3)
    inspectConfig = _messages.MessageField('GooglePrivacyDlpV2InspectConfig', 4)
    locationId = _messages.StringField(5)