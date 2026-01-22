from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ColumnReferencesValueListEntry(_messages.Message):
    """The pair of the foreign key column and primary key column.

      Fields:
        referencedColumn: Required. The column in the primary key that are
          referenced by the referencing_column.
        referencingColumn: Required. The column that composes the foreign key.
      """
    referencedColumn = _messages.StringField(1)
    referencingColumn = _messages.StringField(2)