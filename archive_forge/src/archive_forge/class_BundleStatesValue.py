from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class BundleStatesValue(_messages.Message):
    """The state of the any bundles included in the chosen version of the
    manifest

    Messages:
      AdditionalProperty: An additional property for a BundleStatesValue
        object.

    Fields:
      additionalProperties: Additional properties of type BundleStatesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a BundleStatesValue object.

      Fields:
        key: Name of the additional property.
        value: A PolicyControllerOnClusterState attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('PolicyControllerOnClusterState', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)