from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SchemaFieldSpec(_messages.Message):
    """JSON template for FieldSpec resource for Schemas in Directory API.

  Messages:
    NumericIndexingSpecValue: Indexing spec for a numeric field. By default,
      only exact match queries will be supported for numeric fields. Setting
      the numericIndexingSpec allows range queries to be supported.

  Fields:
    displayName: Display Name of the field.
    etag: ETag of the resource.
    fieldId: Unique identifier of Field (Read-only)
    fieldName: Name of the field.
    fieldType: Type of the field.
    indexed: Boolean specifying whether the field is indexed or not.
    kind: Kind of resource this is.
    multiValued: Boolean specifying whether this is a multi-valued field or
      not.
    numericIndexingSpec: Indexing spec for a numeric field. By default, only
      exact match queries will be supported for numeric fields. Setting the
      numericIndexingSpec allows range queries to be supported.
    readAccessType: Read ACLs on the field specifying who can view values of
      this field. Valid values are "ALL_DOMAIN_USERS" and "ADMINS_AND_SELF".
  """

    class NumericIndexingSpecValue(_messages.Message):
        """Indexing spec for a numeric field.

    By default, only exact match
    queries will be supported for numeric fields. Setting the
    numericIndexingSpec allows range queries to be supported.

    Fields:
      maxValue: Maximum value of this field. This is meant to be indicative
        rather than enforced. Values outside this range will still be indexed,
        but search may not be as performant.
      minValue: Minimum value of this field. This is meant to be indicative
        rather than enforced. Values outside this range will still be indexed,
        but search may not be as performant.
    """
        maxValue = _messages.FloatField(1)
        minValue = _messages.FloatField(2)
    displayName = _messages.StringField(1)
    etag = _messages.StringField(2)
    fieldId = _messages.StringField(3)
    fieldName = _messages.StringField(4)
    fieldType = _messages.StringField(5)
    indexed = _messages.BooleanField(6, default=True)
    kind = _messages.StringField(7, default=u'admin#directory#schema#fieldspec')
    multiValued = _messages.BooleanField(8)
    numericIndexingSpec = _messages.MessageField('NumericIndexingSpecValue', 9)
    readAccessType = _messages.StringField(10, default=u'ALL_DOMAIN_USERS')