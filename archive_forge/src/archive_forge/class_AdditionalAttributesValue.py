from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class AdditionalAttributesValue(_messages.Message):
    """The additional searchable attributes of this resource. The attributes
    may vary from one resource type to another. Examples: `projectId` for
    Project, `dnsName` for DNS ManagedZone. This field contains a subset of
    the resource metadata fields that are returned by the List or Get APIs
    provided by the corresponding Google Cloud service (e.g., Compute Engine).
    see [API references and supported searchable
    attributes](https://cloud.google.com/asset-inventory/docs/supported-asset-
    types) to see which fields are included. You can search values of these
    fields through free text search. However, you should not consume the field
    programically as the field names and values may change as the Google Cloud
    service updates to a new incompatible API version. To search against the
    `additional_attributes`: * Use a free text query to match the attributes
    values. Example: to search `additional_attributes = { dnsName: "foobar"
    }`, you can issue a query `foobar`.

    Messages:
      AdditionalProperty: An additional property for a
        AdditionalAttributesValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a AdditionalAttributesValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('extra_types.JsonValue', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)