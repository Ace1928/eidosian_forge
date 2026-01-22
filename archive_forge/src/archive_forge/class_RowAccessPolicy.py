from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RowAccessPolicy(_messages.Message):
    """Represents access on a subset of rows on the specified table, defined by
  its filter predicate. Access to the subset of rows is controlled by its IAM
  policy.

  Fields:
    creationTime: Output only. The time when this row access policy was
      created, in milliseconds since the epoch.
    etag: Output only. A hash of this resource.
    filterPredicate: Required. A SQL boolean expression that represents the
      rows defined by this row access policy, similar to the boolean
      expression in a WHERE clause of a SELECT query on a table. References to
      other tables, routines, and temporary functions are not supported.
      Examples: region="EU" date_field = CAST('2019-9-27' as DATE)
      nullable_field is not NULL numeric_field BETWEEN 1.0 AND 5.0
    lastModifiedTime: Output only. The time when this row access policy was
      last modified, in milliseconds since the epoch.
    rowAccessPolicyReference: Required. Reference describing the ID of this
      row access policy.
  """
    creationTime = _messages.StringField(1)
    etag = _messages.StringField(2)
    filterPredicate = _messages.StringField(3)
    lastModifiedTime = _messages.StringField(4)
    rowAccessPolicyReference = _messages.MessageField('RowAccessPolicyReference', 5)