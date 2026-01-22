from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V1Beta1QuotaBucket(_messages.Message):
    """A quota bucket is a quota provisioning unit for a specific set of
  dimensions.

  Messages:
    DimensionsValue: The dimensions of this quota bucket.  If this map is
      empty, this is the global bucket, which is the default quota value
      applied to all requests that do not have a more specific override.  If
      this map is nonempty, the default limit, effective limit, and quota
      overrides apply only to requests that have the dimensions given in the
      map.  For example, if the map has key "region" and value "us-east-1",
      then the specified effective limit is only effective in that region, and
      the specified overrides apply only in that region.

  Fields:
    adminOverride: Admin override on this quota bucket.
    consumerOverride: Consumer override on this quota bucket.
    defaultLimit: The default limit of this quota bucket, as specified by the
      service configuration.
    dimensions: The dimensions of this quota bucket.  If this map is empty,
      this is the global bucket, which is the default quota value applied to
      all requests that do not have a more specific override.  If this map is
      nonempty, the default limit, effective limit, and quota overrides apply
      only to requests that have the dimensions given in the map.  For
      example, if the map has key "region" and value "us-east-1", then the
      specified effective limit is only effective in that region, and the
      specified overrides apply only in that region.
    effectiveLimit: The effective limit of this quota bucket. Equal to
      default_limit if there are no overrides.
    producerOverride: Producer override on this quota bucket.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DimensionsValue(_messages.Message):
        """The dimensions of this quota bucket.  If this map is empty, this is
    the global bucket, which is the default quota value applied to all
    requests that do not have a more specific override.  If this map is
    nonempty, the default limit, effective limit, and quota overrides apply
    only to requests that have the dimensions given in the map.  For example,
    if the map has key "region" and value "us-east-1", then the specified
    effective limit is only effective in that region, and the specified
    overrides apply only in that region.

    Messages:
      AdditionalProperty: An additional property for a DimensionsValue object.

    Fields:
      additionalProperties: Additional properties of type DimensionsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DimensionsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    adminOverride = _messages.MessageField('V1Beta1QuotaOverride', 1)
    consumerOverride = _messages.MessageField('V1Beta1QuotaOverride', 2)
    defaultLimit = _messages.IntegerField(3)
    dimensions = _messages.MessageField('DimensionsValue', 4)
    effectiveLimit = _messages.IntegerField(5)
    producerOverride = _messages.MessageField('V1Beta1QuotaOverride', 6)