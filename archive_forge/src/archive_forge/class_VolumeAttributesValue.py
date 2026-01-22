from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class VolumeAttributesValue(_messages.Message):
    """stores driver specific attributes. For Google Cloud Storage volumes,
    the following attributes are supported: * bucketName: the name of the
    Cloud Storage bucket to mount. The Cloud Run Service identity must have
    access to this bucket.

    Messages:
      AdditionalProperty: An additional property for a VolumeAttributesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        VolumeAttributesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a VolumeAttributesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)