from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ColumnFamiliesValue(_messages.Message):
    """The column families configured for this table, mapped by column family
    ID. Views: `SCHEMA_VIEW`, `STATS_VIEW`, `FULL`

    Messages:
      AdditionalProperty: An additional property for a ColumnFamiliesValue
        object.

    Fields:
      additionalProperties: Additional properties of type ColumnFamiliesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ColumnFamiliesValue object.

      Fields:
        key: Name of the additional property.
        value: A ColumnFamily attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('ColumnFamily', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)