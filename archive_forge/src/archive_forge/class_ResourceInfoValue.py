from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ResourceInfoValue(_messages.Message):
    """Cached version of all the metrics of interest for the job. This value
    gets stored here when the job is terminated. As long as the job is
    running, this field is populated from the Dataflow API.

    Messages:
      AdditionalProperty: An additional property for a ResourceInfoValue
        object.

    Fields:
      additionalProperties: Additional properties of type ResourceInfoValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ResourceInfoValue object.

      Fields:
        key: Name of the additional property.
        value: A number attribute.
      """
        key = _messages.StringField(1)
        value = _messages.FloatField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)