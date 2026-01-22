from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VersionedResource(_messages.Message):
    """Resource representation as defined by the corresponding service
  providing the resource for a given API version.

  Messages:
    ResourceValue: JSON representation of the resource as defined by the
      corresponding service providing this resource. Example: If the resource
      is an instance provided by Compute Engine, this field will contain the
      JSON representation of the instance as defined by Compute Engine:
      `https://cloud.google.com/compute/docs/reference/rest/v1/instances`. You
      can find the resource definition for each supported resource type in
      this table: `https://cloud.google.com/asset-inventory/docs/supported-
      asset-types`

  Fields:
    resource: JSON representation of the resource as defined by the
      corresponding service providing this resource. Example: If the resource
      is an instance provided by Compute Engine, this field will contain the
      JSON representation of the instance as defined by Compute Engine:
      `https://cloud.google.com/compute/docs/reference/rest/v1/instances`. You
      can find the resource definition for each supported resource type in
      this table: `https://cloud.google.com/asset-inventory/docs/supported-
      asset-types`
    version: API version of the resource. Example: If the resource is an
      instance provided by Compute Engine v1 API as defined in
      `https://cloud.google.com/compute/docs/reference/rest/v1/instances`,
      version will be "v1".
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ResourceValue(_messages.Message):
        """JSON representation of the resource as defined by the corresponding
    service providing this resource. Example: If the resource is an instance
    provided by Compute Engine, this field will contain the JSON
    representation of the instance as defined by Compute Engine:
    `https://cloud.google.com/compute/docs/reference/rest/v1/instances`. You
    can find the resource definition for each supported resource type in this
    table: `https://cloud.google.com/asset-inventory/docs/supported-asset-
    types`

    Messages:
      AdditionalProperty: An additional property for a ResourceValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ResourceValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    resource = _messages.MessageField('ResourceValue', 1)
    version = _messages.StringField(2)