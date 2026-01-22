from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ProtectedResourcesSummary(_messages.Message):
    """Aggregate information about the resources protected by a Cloud KMS key
  in the same Cloud organization as the key.

  Messages:
    CloudProductsValue: The number of resources protected by the key grouped
      by Cloud product.
    LocationsValue: The number of resources protected by the key grouped by
      region.
    ResourceTypesValue: The number of resources protected by the key grouped
      by resource type.

  Fields:
    cloudProducts: The number of resources protected by the key grouped by
      Cloud product.
    locations: The number of resources protected by the key grouped by region.
    name: The full name of the ProtectedResourcesSummary resource. Example:
      projects/test-project/locations/us/keyRings/test-
      keyring/cryptoKeys/test-key/protectedResourcesSummary
    projectCount: The number of distinct Cloud projects in the same Cloud
      organization as the key that have resources protected by the key.
    resourceCount: The total number of protected resources in the same Cloud
      organization as the key.
    resourceTypes: The number of resources protected by the key grouped by
      resource type.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class CloudProductsValue(_messages.Message):
        """The number of resources protected by the key grouped by Cloud product.

    Messages:
      AdditionalProperty: An additional property for a CloudProductsValue
        object.

    Fields:
      additionalProperties: Additional properties of type CloudProductsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a CloudProductsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.IntegerField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LocationsValue(_messages.Message):
        """The number of resources protected by the key grouped by region.

    Messages:
      AdditionalProperty: An additional property for a LocationsValue object.

    Fields:
      additionalProperties: Additional properties of type LocationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LocationsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.IntegerField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ResourceTypesValue(_messages.Message):
        """The number of resources protected by the key grouped by resource type.

    Messages:
      AdditionalProperty: An additional property for a ResourceTypesValue
        object.

    Fields:
      additionalProperties: Additional properties of type ResourceTypesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ResourceTypesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.IntegerField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    cloudProducts = _messages.MessageField('CloudProductsValue', 1)
    locations = _messages.MessageField('LocationsValue', 2)
    name = _messages.StringField(3)
    projectCount = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    resourceCount = _messages.IntegerField(5)
    resourceTypes = _messages.MessageField('ResourceTypesValue', 6)