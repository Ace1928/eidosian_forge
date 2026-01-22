from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class AllocationsValue(_messages.Message):
    """Mapping from version IDs within the service to fractional (0.000, 1]
    allocations of traffic for that version. Each version can be specified
    only once, but some versions in the service may not have any traffic
    allocation. Services that have traffic allocated cannot be deleted until
    either the service is deleted or their traffic allocation is removed.
    Allocations must sum to 1. Up to two decimal place precision is supported
    for IP-based splits and up to three decimal places is supported for
    cookie-based splits.

    Messages:
      AdditionalProperty: An additional property for a AllocationsValue
        object.

    Fields:
      additionalProperties: Additional properties of type AllocationsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a AllocationsValue object.

      Fields:
        key: Name of the additional property.
        value: A number attribute.
      """
        key = _messages.StringField(1)
        value = _messages.FloatField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)